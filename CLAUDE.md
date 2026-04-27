# CLAUDE.md

## Project Overview
This repo contains code for a SMILES-based variational autoencoder (VAE).

### SMILES Tokenizer
The tokenizer is implemented in `code/smiles_tokenizer.py` as a standalone class with minimal dependencies (re, numpy). It replaces the previous DeepChem dependency.

**Class:** `SMILESTokenizer(vocab_file="data/vocab.txt")`

**Methods:**
- `encode(smiles)` - Converts SMILES string(s) to token IDs. Returns numpy array (1D for single string, 2D for list).
- `decode(token_ids)` - Converts token IDs back to SMILES string(s).
- `add_padding_tokens(token_ids, max_length)` - Pads token ID sequence to max_length.

**Example:**
```python
from code.smiles_tokenizer import SMILESTokenizer

tokenizer = SMILESTokenizer("data/vocab.txt")
encoded = tokenizer.encode("CCO")  # Returns: numpy array [12, 16, 16, 19, 13]
decoded = tokenizer.decode(encoded)  # Returns: "CCO"
padded = tokenizer.add_padding_tokens(encoded, max_length=20)  # Padded to length 20
``` 

## Tech Stack
- Language: Python
- Vocab file: 'data/vocab.txt'
- packages:
  * RDKit for general utility
  * Numpy for arrays
  * re for regular expressions (tokenizer)
  * TensorFlow/Keras for VAE model

## Project Structure
```
code/
  smiles_vae.py           # VAE model implementation
  smiles_tokenizer.py     # SMILES tokenizer class
data/
  vocab.txt               # Vocabulary file (591 tokens)
```

## Other information
- assume any API keys can be obtained as environment variables.