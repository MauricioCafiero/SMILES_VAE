"""
Driver script that mirrors the SMILES_VAE.ipynb notebook, adapted to run
locally (the notebook was written for Colab, where the repo is cloned into a
`SMILES_VAE/` subfolder and the dataset lives at `/content/`).

Run from the repo root:

    .venv/bin/python code/run_train.py [--nrows 3000] [--epochs 2] [--generate 50]

What it does (matches the notebook cells):
  1. Load data/ZN15_55K.csv and strip salt fragments (same cleaning as
     smiles_vae.make_datasets).
  2. Tokenize with SMILESTokenizer(data/vocab_305K.txt), pad to the longest
     SMILES, and build teacher-forcing (x, y) pairs.
  3. Build and train the VAE with the notebook's hyperparameters
     (emb=512, latent=512, 1 GRU layer, 256 units, scale_ll=10.0).
  4. Reconstruct 20 training molecules with test_vae and save the grid image.
  5. Generate 50 novel molecules from the latent space and save the grid
     image plus the raw SMILES list.

By default it subsamples the dataset to 3000 rows and trains for 2 epochs so
the whole thing finishes in a few minutes on CPU. Pass --nrows 0 to use the
full 55k rows (slow on CPU).
"""
import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

# Make code/ importable so `from smiles_vae import VAE` works regardless of CWD.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rdkit import Chem  # noqa: E402
from smiles_tokenizer import SMILESTokenizer  # noqa: E402
from smiles_vae import VAE  # noqa: E402

# Reproducibility: seed Python/numpy/TF, and try to make ops deterministic.
# This removes the run-to-run variance seen on 2026-07-25 (recon 0.50 vs 0.55
# from identical configs). enable_op_determinism can fail on some ops / builds
# (notably cuDNN GRU on GPU, where it can raise at fit time), so it's guarded
# and can be disabled with SMILES_VAE_DISABLE_OP_DETERMINISM=1. The seed alone
# still buys most of the stability.
SEED = 42
import random as _random
_random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
if os.environ.get("SMILES_VAE_DISABLE_OP_DETERMINISM") == "1":
    print("[seed] op determinism disabled by env var; using seed only.")
else:
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        print(f"[seed] enable_op_determinism unavailable ({e}); continuing with seed only.")


def save_grid(mols, legends, path, mols_per_row=2):
    """Render a molecule grid image, robust to RDKit API changes across versions.

    Newer RDKit (2026.x) changed MolDraw2D.DrawMolecules' signature so the
    notebook's MolsToGridImage call raises Boost.Python.ArgumentError. Fall
    back to drawing each mol into its own PNG and tiling them manually with
    PIL so a rendering hiccup never kills the training run.
    """
    from rdkit.Chem import Draw
    from PIL import Image

    valid = [(m, l) for m, l in zip(mols, legends) if m is not None]
    if not valid:
        return False
    mols_ok, legends_ok = zip(*valid)
    try:
        img = Draw.MolsToGridImage(
            mols=list(mols_ok), legends=list(legends_ok),
            molsPerRow=mols_per_row, maxMols=len(mols_ok))
        img.save(path)
        return True
    except Exception:
        # Fallback: individual molecule images tiled with PIL.
        size = (200, 200)
        tiles = [Draw.MolToImage(m, size=size, legend=l)
                 for m, l in zip(mols_ok, legends_ok)]
        cols = mols_per_row
        rows = (len(tiles) + cols - 1) // cols
        canvas = Image.new("RGB", (cols * size[0], rows * size[1]), "white")
        for i, t in enumerate(tiles):
            canvas.paste(t, ((i % cols) * size[0], (i // cols) * size[1]))
        canvas.save(path)
        return True


def decode_latents(vae, tokenizer, z_gen, temperature=0.0):
    """Map a batch of latent vectors to SMILES via the autoregressive decoder.

    Returns (smiles_list, valid_pairs) where valid_pairs is a list of
    (rdkit.Mol, smiles) for the chemically valid, non-degenerate outputs.
    temperature: 0.0 = greedy argmax; >0 = Gumbel-max sampling (diversity).
    """
    smiles, _ = vae.autoregressive_decode(z_gen, tokenizer, temperature)
    # Honest validity: skip empty strings (Chem.MolFromSmiles("") is non-None
    # in this RDKit build, which inflated earlier counts) and require a real,
    # non-degenerate molecule (>1 atom). This is the metric we tune against.
    valid = []
    for s in smiles:
        if not s:
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is not None and mol.GetNumAtoms() > 1:
            valid.append((mol, s))
    return smiles, valid


def empirical_latent_stats(vae):
    """Estimate the latent distribution from the training set's z_mean.

    With a low KL weight the aggregate posterior is NOT N(0, I), so sampling
    z ~ N(0, I) puts points off the data manifold. Sampling from the
    empirical per-dimension mean/std instead lands on-manifold far more often.
    """
    enc_out = vae.encoder.predict(vae.X, verbose=0)
    z_mean = enc_out[0]  # encoder outputs [z_mean, z_log_var, z]
    mu = z_mean.mean(axis=0)
    sigma = z_mean.std(axis=0) + 1e-6
    return mu, sigma

class KLAnnealingCallback(tf.keras.callbacks.Callback):
    """Linearly ramp the KL loss weight from 0 -> target over anneal_epochs.

    Posterior collapse: with teacher forcing the decoder can predict next tokens
    from the input alone and ignore z. Annealing the KL up from zero lets the
    model first learn reconstruction, then gradually commit z to be informative.
    KL_Loss_Layer reads its `scale_ll` attribute on every forward pass, so
    mutating it on_epoch_begin takes effect for the next epoch.
    """
    def __init__(self, kl_layer, target, anneal_epochs):
        super().__init__()
        self.kl_layer = kl_layer
        self.target = float(target)
        self.anneal_epochs = int(anneal_epochs)

    def on_epoch_begin(self, epoch, logs=None):
        if self.anneal_epochs <= 0:
            w = self.target
        else:
            w = self.target * min(1.0, epoch / self.anneal_epochs)
        self.kl_layer.scale_ll = w
        print(f"[anneal] epoch {epoch + 1}: scale_ll = {w:.5f}")

REPO_ROOT = os.path.dirname(HERE)
DATA_CSV = os.path.join(REPO_ROOT, "data", "ZN15_55K.csv")
VOCAB_FILE = os.path.join(REPO_ROOT, "data", "vocab_305K.txt")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")


def clean_salts(smiles: str) -> str:
    """Strip common salt fragments, matching smiles_vae.make_datasets."""
    for pat in ("[Na+].", "[Cl-].", ".[Cl-]", ".[Na+]",
                "[K+].", "[Br-].", ".[K+]", ".[Br-]",
                "[I-].", ".[I-]", "[Ca2+].", ".[Ca2+]"):
        smiles = smiles.replace(pat, "")
    return smiles


def build_dataset(nrows):
    """Tokenize the CSV and return (X, y, vocab_size, tokenizer, max_length, smiles_list)."""
    df = pd.read_csv(DATA_CSV)
    if nrows and nrows > 0:
        df = df.sample(n=min(nrows, len(df)), random_state=42).reset_index(drop=True)

    smiles_list = [clean_salts(s) for s in df["SMILES"]]

    tokenizer = SMILESTokenizer(vocab_file=VOCAB_FILE)

    encoded = [tokenizer.encode(s) for s in smiles_list]
    biggest = max(len(e) for e in encoded)
    max_length = biggest

    padded = [tokenizer.add_padding_tokens(e, max_length) for e in encoded]

    # Teacher forcing: predict next token from previous tokens.
    x = np.array([p[0:max_length - 1] for p in padded], dtype=np.int32)
    y = np.array([p[1:max_length] for p in padded], dtype=np.int32)

    with open(VOCAB_FILE) as f:
        vocab_size = len(f.readlines())

    print(f"Dataset rows: {len(smiles_list)}")
    print(f"Max token length: {max_length}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"X shape: {x.shape}, y shape: {y.shape}")
    return x, y, vocab_size, tokenizer, max_length, smiles_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrows", type=int, default=3000,
                    help="rows to sample (0 = full dataset). default 3000")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--generate", type=int, default=50,
                    help="number of novel molecules to generate")
    ap.add_argument("--emb", type=int, default=512, help="embedding size")
    ap.add_argument("--latent", type=int, default=512, help="latent size")
    ap.add_argument("--units", type=int, default=256, help="GRU units per layer")
    ap.add_argument("--layers", type=int, default=1, help="GRU layers")
    ap.add_argument("--scale_ll", type=float, default=10.0,
                    help="KL loss weight (main latent-structure lever)")
    ap.add_argument("--word_dropout_keep", type=float, default=0.8,
                    help="fraction of decoder input tokens kept during training "
                         "(1.0 = off; lower to force latent z usage, Bowman 2016)")
    ap.add_argument("--anneal_epochs", type=int, default=10,
                    help="linearly ramp scale_ll 0 -> target over this many epochs "
                         "(0 = no annealing)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="decode sampling temperature (0.0 = greedy argmax; "
                         "0.7-1.0 = diverse Gumbel-max sampling)")
    ap.add_argument("--load", type=str, default="",
                    help="path to a saved run dir to load weights from "
                         "(skips training; re-runs reconstruction + generation)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    x, y, vocab_size, tokenizer, max_length, smiles_list = build_dataset(args.nrows)

    # Hyperparameters (defaults match the notebook's `new_VAE` cell).
    print(f"Config: emb={args.emb} latent={args.latent} units={args.units} "
          f"layers={args.layers} scale_ll={args.scale_ll} "
          f"word_dropout_keep={args.word_dropout_keep} anneal_epochs={args.anneal_epochs}")
    vae = VAE(emb_size=args.emb, latent_size=args.latent, num_layers=args.layers,
              num_units=args.units, scale_ll=args.scale_ll,
              max_length=max_length - 1, vocab_size=vocab_size,
              cls_id=tokenizer.cls_token_id, sep_id=tokenizer.sep_token_id,
              pad_id=tokenizer.pad_token_id, unk_id=tokenizer.unk_token_id,
              word_dropout_keep=args.word_dropout_keep)
    vae.make_vae()
    vae.compile_vae(x, y, epochs=args.epochs)

    # Per-run output dir grouping all artifacts (weights, config, history,
    # generated SMILES) so a trained model can be re-loaded with --load <dir>.
    cfg = (f"n{args.nrows}_e{args.epochs}_b32_emb{args.emb}_lat{args.latent}"
           f"_gru{args.units}_ll{args.scale_ll}_wd{args.word_dropout_keep}"
           f"_ann{args.anneal_epochs}")
    run_dir = os.path.join(OUT_DIR, f"run_{cfg}_{datetime.now():%Y%m%d_%H%M%S}")
    enc_w = os.path.join(run_dir, "encoder.weights.h5")
    dec_w = os.path.join(run_dir, "decoder.weights.h5")
    cfg_path = os.path.join(run_dir, "config.json")
    ckpt_w = os.path.join(run_dir, "ckpt_best.weights.h5")

    if args.load:
        # Re-use a previously trained model: load weights, skip training.
        load_dir = args.load
        print(f"\n=== Loading weights from {load_dir} (skipping training) ===")
        vae.encoder.load_weights(os.path.join(load_dir, "encoder.weights.h5"))
        vae.decoder.load_weights(os.path.join(load_dir, "decoder.weights.h5"))
        history = None
    else:
        os.makedirs(run_dir, exist_ok=True)
        # Checkpoint the best (val_loss) autoencoder weights during training.
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_w, monitor="val_loss", save_best_only=True,
            save_weights_only=True, verbose=1)
        # KL annealing: ramp scale_ll 0 -> target over --anneal_epochs.
        anneal_cb = KLAnnealingCallback(vae.kl_layer, args.scale_ll, args.anneal_epochs)
        print("\n=== Training ===")
        history = vae.train_vae(callbacks=[checkpoint_cb, anneal_cb])

        # Restore best weights (if a checkpoint was saved) before persisting.
        if os.path.exists(ckpt_w):
            print(f"Restoring best checkpoint -> {ckpt_w}")
            vae.autoencoder.load_weights(ckpt_w)
        vae.encoder.save_weights(enc_w)
        vae.decoder.save_weights(dec_w)
        import json
        with open(cfg_path, "w") as f:
            json.dump({
                "emb": args.emb, "latent": args.latent, "units": args.units,
                "layers": args.layers, "scale_ll": args.scale_ll,
                "word_dropout_keep": args.word_dropout_keep,
                "anneal_epochs": args.anneal_epochs,
                "temperature": args.temperature,
                "max_length": max_length - 1, "vocab_size": vocab_size,
                "nrows": args.nrows, "epochs": args.epochs,
            }, f, indent=2)
        print(f"Saved encoder weights -> {enc_w}")
        print(f"Saved decoder weights -> {dec_w}")
        print(f"Saved config          -> {cfg_path}")

    # Per-epoch history — the tuning signal (skipped when loading a model).
    if history is not None:
        h = history.history
        print("\n=== History ===")
        print(f"{'epoch':>5} {'loss':>10} {'val_loss':>10} {'acc':>8} {'val_acc':>8}")
        for ep in range(len(h["loss"])):
            print(f"{ep+1:>5} {h['loss'][ep]:>10.4f} {h['val_loss'][ep]:>10.4f} "
                  f"{h['accuracy'][ep]:>8.4f} {h['val_accuracy'][ep]:>8.4f}")

        # Save history to CSV for cross-run comparison while tuning hyperparams.
        hist_path = os.path.join(run_dir, "history.csv")
        with open(hist_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "loss", "val_loss", "accuracy", "val_accuracy"])
            for ep in range(len(h["loss"])):
                w.writerow([ep + 1, h["loss"][ep], h["val_loss"][ep],
                            h["accuracy"][ep], h["val_accuracy"][ep]])
        print(f"Saved history -> {hist_path}")

    # Outputs (grids, generated SMILES) go into the loaded run dir on --load,
    # otherwise into this run's fresh dir.
    out_dir = args.load if args.load else run_dir
    os.makedirs(out_dir, exist_ok=True)

    # ---- Reconstruction test: encode -> z_mean -> autoregressive decode ----
    # This is the honest "does z carry enough to reconstruct?" test (the whole
    # point of a VAE), not the old single-pass copy-the-input reconstruction.
    print("\n=== Reconstruction test (20 molecules, from z_mean) ===")
    test_size = 20
    rng = np.random.RandomState(0)
    idx = rng.randint(0, len(smiles_list), size=test_size)
    raw = [smiles_list[i] for i in idx]
    enc = np.array([tokenizer.add_padding_tokens(tokenizer.encode(s), vae.max_length) for s in raw])
    z_mean = vae.encoder.predict(enc, verbose=0)[0]   # encoder outputs [z_mean, z_log_var, z]
    recon, _ = vae.autoregressive_decode(z_mean, tokenizer)

    hits = sum(1 for o, n in zip(raw, recon) if o == n)
    print(f"Hits: {hits}")
    print(f"Losses: {test_size - hits}")
    print(f"Accuracy: {hits / test_size}")

    # Diagnostic: is z actually informative about the *specific* input? Exact
    # string match is too strict (0/20 doesn't distinguish "z uninformative"
    # from "z informative but greedy decode gives a valid variant"). So also
    # report per-reconstruction validity + Tanimoto fingerprint similarity to
    # the input, and dump input->reconstruction pairs to a text file we can
    # read without viewing the PNG image.
    from rdkit.Chem import AllChem, DataStructs
    def _fp(s):
        m = Chem.MolFromSmiles(s)
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None
    sim_scores, recon_valid = [], 0
    rows = []
    for o, n in zip(raw, recon):
        n_mol = Chem.MolFromSmiles(n)
        n_valid = bool(n_mol is not None and n_mol.GetNumAtoms() > 1)
        recon_valid += int(n_valid)
        fp_o, fp_n = _fp(o), _fp(n) if n_mol else None
        sim = DataStructs.TanimotoSimilarity(fp_o, fp_n) if (fp_o and fp_n) else 0.0
        sim_scores.append(sim)
        rows.append(f"{o}\t{n}\t{'valid' if n_valid else 'invalid'}\t{sim:.3f}")
    mean_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0.0
    print(f"Recon validity: {recon_valid}/{test_size}")
    print(f"Mean Tanimoto(input, recon): {mean_sim:.3f}  "
          f"(1.0 = identical fingerprint, 0.0 = no shared substructure)")
    recon_txt = os.path.join(out_dir, "reconstruction_smiles.txt")
    with open(recon_txt, "w") as f:
        f.write("# input\treconstruction\tvalid\ttanimoto\n")
        f.write(f"# mean_tanimoto={mean_sim:.3f} recon_valid={recon_valid}/{test_size}\n")
        for r in rows:
            f.write(r + "\n")
    print(f"Saved reconstruction pairs -> {recon_txt}")

    mols, legends = [], []
    for o, n in zip(raw, recon):
        mols.append(Chem.MolFromSmiles(o))
        mols.append(Chem.MolFromSmiles(n))
        legends.append("Input")
        legends.append("Reconstruction")
    recon_grid = os.path.join(out_dir, "reconstruction_grid.png")
    if save_grid(mols, legends, recon_grid):
        print(f"Saved reconstruction grid -> {recon_grid}")

    # ---- Novel molecule generation ----
    # Compare two sampling strategies: standard N(0, I) vs empirical latent
    # (mean/std of the training set's z_mean). The latter lands on-manifold.
    print(f"\n=== Novel molecule generation ({args.generate} samples) ===")

    # Standard: z ~ N(0, I)  (what VAE.generate does)
    z_std = np.random.normal(size=(args.generate, vae.latent_size))
    std_smiles, std_valid = decode_latents(vae, tokenizer, z_std, args.temperature)
    print(f"[standard  N(0,I)] valid: {len(std_valid)}/{len(std_smiles)} "
          f"({len(std_valid)/len(std_smiles):.2%})")

    # Empirical: z ~ mu + sigma * N(0, I), learned from the training latents
    mu, sigma = empirical_latent_stats(vae)
    z_emp = mu + sigma * np.random.normal(size=(args.generate, vae.latent_size))
    emp_smiles, emp_valid = decode_latents(vae, tokenizer, z_emp, args.temperature)
    print(f"[empirical mu/sigma] valid: {len(emp_valid)}/{len(emp_smiles)} "
          f"({len(emp_valid)/len(emp_smiles):.2%})")

    # Use whichever strategy produced more valid molecules for the saved grid.
    best_smiles, best_valid, label = (
        (emp_smiles, emp_valid, "empirical") if len(emp_valid) >= len(std_valid)
        else (std_smiles, std_valid, "standard"))
    vae.new_mols = best_smiles
    print(f"Best strategy: {label} ({len(best_valid)} valid)")
    if best_valid:
        gmols, gleg = zip(*best_valid)
        gen_grid = os.path.join(out_dir, "generated_grid.png")
        if save_grid(list(gmols), list(gleg), gen_grid):
            print(f"Saved generated grid -> {gen_grid}")

    # Persist both sets for inspection.
    for tag, smiles in (("standard", std_smiles), ("empirical", emp_smiles)):
        out_smiles = os.path.join(out_dir, f"generated_smiles_{tag}.txt")
        with open(out_smiles, "w") as f:
            for sm in smiles:
                f.write(sm + "\n")
        print(f"Saved {len(smiles)} {tag} SMILES -> {out_smiles}")
    print("\nDone.")


if __name__ == "__main__":
    main()