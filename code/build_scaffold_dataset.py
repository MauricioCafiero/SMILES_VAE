"""Build a Murcko-scaffold-balanced, length-capped SMILES dataset.

Motivation: on the ZN15_55K dataset the VAE collapsed non-lipid inputs to long
alkane/fatty chains. Two drivers: (1) the long lipid sequences dominate the
per-token cross-entropy (token-mass imbalance), and (2) no scaffold/class
balance, so the decoder's prior defaults to the majority class. This script
counteracts both by capping how many molecules each Murcko scaffold may
contribute and by filtering out very long SMILES, then drawing a fixed-size
representative sample from the capped pool.

Pipeline:
  1. Read a one-column SMILES CSV.
  2. Parse with RDKit; drop unparseable.
  3. Token-length filter (via SMILESTokenizer) — drops the long chains that
     dominate the loss, keeping the bulk of drug-like molecules.
  4. Compute the Murcko scaffold for each molecule; group by scaffold SMILES
     (acyclic molecules share the empty-string scaffold, so they get capped
     just like any other group).
  5. Cap each scaffold group at --max-per-scaffold (the dominant classes are
     trimmed; rare scaffolds keep all their members).
  6. Uniformly sample --target-n molecules from the capped pool (seeded).
  7. Write a one-column SMILES CSV ready for run_train.py.

Run from the repo root:

    .venv/bin/python code/build_scaffold_dataset.py \
        --input ZN305K_smiles.csv \
        --output data/ZN305K_scaffold_balanced_50k.csv \
        --max-tokens 100 --max-per-scaffold 20 --target-n 50000
"""
import argparse
import csv
import os
import random
import sys
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rdkit import Chem  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402
from smiles_tokenizer import SMILESTokenizer  # noqa: E402

REPO_ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=os.path.join(REPO_ROOT, "ZN305K_smiles.csv"),
                    help="input one-column SMILES CSV (default: repo-root ZN305K_smiles.csv)")
    ap.add_argument("--output", default=os.path.join(REPO_ROOT, "data", "ZN305K_scaffold_balanced_50k.csv"),
                    help="output one-column SMILES CSV")
    ap.add_argument("--smiles-column", default="SMILES",
                    help="column name / header for the SMILES in --input")
    ap.add_argument("--max-tokens", type=int, default=100,
                    help="drop molecules whose tokenized length exceeds this "
                         "(removes long-chain token-mass dominance; default 100)")
    ap.add_argument("--max-per-scaffold", type=int, default=20,
                    help="cap molecules per Murcko scaffold (default 20)")
    ap.add_argument("--target-n", type=int, default=50000,
                    help="number of molecules to sample from the capped pool (0 = keep all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # --- load ---
    with open(args.input) as f:
        rows = list(csv.reader(f))
    header, data = rows[0], [r for r in rows[1:] if r]
    # Support both one-column files and multi-column with a named SMILES column.
    if len(header) == 1:
        col = 0
    elif args.smiles_column in header:
        col = header.index(args.smiles_column)
    else:
        col = 0
    smiles = [r[col] for r in data]
    print(f"Input: {len(smiles)} SMILES from {args.input}")

    tokenizer = SMILESTokenizer(
        vocab_file=os.path.join(REPO_ROOT, "data", "vocab_305K.txt"))

    # --- parse + length filter + scaffold ---
    invalid = 0
    too_long = 0
    groups = collections.defaultdict(list)   # scaffold SMILES -> [smiles, ...]
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            invalid += 1
            continue
        try:
            tl = len(tokenizer.encode(s))
        except Exception:
            invalid += 1
            continue
        if tl > args.max_tokens:
            too_long += 1
            continue
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            scaf_smiles = Chem.MolToSmiles(scaf)
        except Exception:
            scaf_smiles = ""
        groups[scaf_smiles].append(s)
        if (i + 1) % 50000 == 0:
            print(f"  ...processed {i + 1}")

    after_filter = sum(len(v) for v in groups.values())
    print(f"Dropped: {invalid} unparseable, {too_long} over {args.max_tokens} tokens")
    print(f"After length filter: {after_filter} molecules across "
          f"{len(groups)} scaffolds "
          f"({sum(1 for g in groups if g == '')} of which is the empty/acyclic group)")

    # --- cap per scaffold ---
    capped = []
    for scaf, members in groups.items():
        if len(members) > args.max_per_scaffold:
            # Keep the first cap deterministically (input order), then the pool
            # is shuffled before sampling below.
            capped.extend(members[:args.max_per_scaffold])
        else:
            capped.extend(members)
    print(f"After cap {args.max_per_scaffold}/scaffold: {len(capped)} molecules")

    # --- sample target-n from the capped pool ---
    if args.target_n and args.target_n > 0 and len(capped) > args.target_n:
        sample = random.sample(capped, args.target_n)
    else:
        sample = capped
    print(f"Final dataset: {len(sample)} molecules")

    # --- report balance of the final set ---
    final_groups = collections.Counter()
    for s in sample:
        mol = Chem.MolFromSmiles(s)
        scaf = ""
        if mol is not None:
            try:
                scaf = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
            except Exception:
                pass
        final_groups[scaf] += 1
    sizes = sorted(final_groups.values(), reverse=True)
    print("\nFinal scaffold balance:")
    print(f"  unique scaffolds: {len(final_groups)}")
    print(f"  largest scaffold: {sizes[0]} (acyclic/empty: {final_groups.get('', 0)})")
    print(f"  top 5: {sizes[:5]}")
    print("  most common scaffolds:")
    for sc, c in final_groups.most_common(5):
        print(f"    {c:5d}  {sc[:80] or '<acyclic>'}")

    # --- write ---
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([args.smiles_column])
        for s in sample:
            w.writerow([s])
    print(f"\nWrote {len(sample)} SMILES -> {args.output}")


if __name__ == "__main__":
    main()