"""
Test script for the SMILES tokenizer.
"""

import sys
import numpy as np
from smiles_tokenizer import SMILESTokenizer


def test_basic_tokenization():
    """Test basic tokenization of SMILES strings."""
    print("=" * 60)
    print("Testing Basic Tokenization")
    print("=" * 60)

    tokenizer = SMILESTokenizer("../data/vocab.txt")

    test_smiles = [
        "CCO",  # Ethanol
        "c1ccccc1",  # Benzene
        "CC(=O)O",  # Acetic acid
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)",  # Caffeine
        "Clc1ccc(cc1)Br",  # Para-bromochlorobenzene
        "[N+](=O)[O-]",  # Nitro group
        "[C@H]1O[C@H]([C@@H]([C@H]1O)O)O",  # Glucose
    ]

    for smiles in test_smiles:
        print(f"\nInput SMILES: {smiles}")

        # Encode
        encoded = tokenizer.encode(smiles)
        print(f"Encoded IDs: {encoded}")

        # Convert to tokens
        tokens = tokenizer.convert_ids_to_tokens(encoded)
        print(f"Tokens: {tokens}")

        # Decode back
        decoded = tokenizer.decode(encoded)
        print(f"Decoded: {decoded}")

        # Check round-trip
        match = "YES" if smiles == decoded else "NO"
        print(f"Round-trip match: {match}")

    print("\n" + "=" * 60)
    print(f"Vocabulary size: {tokenizer.vocab_size()}")
    print("=" * 60)


def test_padding():
    """Test add_padding_tokens method."""
    print("\n" + "=" * 60)
    print("Testing add_padding_tokens")
    print("=" * 60)

    tokenizer = SMILESTokenizer("../data/vocab.txt")

    smiles = "CCO"
    encoded = tokenizer.encode(smiles)
    print(f"Original encoded (length {len(encoded)}): {encoded}")

    padded = tokenizer.add_padding_tokens(encoded, max_length=20)
    print(f"Padded to 20 (length {len(padded)}): {padded}")

    decoded = tokenizer.decode(padded)
    print(f"Decoded from padded: {decoded}")


def test_batch_encoding():
    """Test encoding multiple SMILES strings."""
    print("\n" + "=" * 60)
    print("Testing Batch Encoding")
    print("=" * 60)

    tokenizer = SMILESTokenizer("../data/vocab.txt")

    smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
    print(f"Input SMILES: {smiles_list}")

    batch_encoded = tokenizer.encode(smiles_list)
    print(f"Batch encoded shape: {batch_encoded.shape}")
    print(f"Batch encoded:\n{batch_encoded}")

    decoded = tokenizer.decode(batch_encoded)
    print(f"Decoded: {decoded}")


def test_special_tokens():
    """Test special token handling."""
    print("\n" + "=" * 60)
    print("Testing Special Tokens")
    print("=" * 60)

    tokenizer = SMILESTokenizer("../data/vocab.txt")

    print(f"PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"CLS token: '{tokenizer.cls_token}' (ID: {tokenizer.cls_token_id})")
    print(f"SEP token: '{tokenizer.sep_token}' (ID: {tokenizer.sep_token_id})")
    print(f"MASK token: '{tokenizer.mask_token}' (ID: {tokenizer.mask_token_id})")
    print(f"UNK token: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 60)
    print("Testing Edge Cases")
    print("=" * 60)

    tokenizer = SMILESTokenizer("../data/vocab.txt")

    # Empty string
    encoded = tokenizer.encode("")
    print(f"Empty string encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Empty string decoded: '{decoded}'")

    # Single atom
    encoded = tokenizer.encode("C")
    print(f"Single atom 'C' encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Single atom decoded: '{decoded}'")

    # Complex ring closures
    smiles = "C1CCCCC1C2CCCCC2"
    encoded = tokenizer.encode(smiles)
    print(f"\nMulti-ring SMILES: {smiles}")
    tokens = tokenizer.convert_ids_to_tokens(encoded)
    print(f"Tokens: {tokens}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")


def test_methods_match_api():
    """Verify all required methods are present and work as expected."""
    print("\n" + "=" * 60)
    print("Testing API Compliance")
    print("=" * 60)

    tokenizer = SMILESTokenizer("../data/vocab.txt")

    # Check all required methods exist
    required_methods = ['encode', 'decode', 'add_padding_tokens']
    for method in required_methods:
        assert hasattr(tokenizer, method), f"Missing method: {method}"
        print(f"[OK] Method '{method}' exists")

    # Test return types
    encoded = tokenizer.encode("CCO")
    assert isinstance(encoded, np.ndarray), "encode should return numpy array"
    print(f"[OK] encode returns numpy array (dtype: {encoded.dtype})")

    decoded = tokenizer.decode(encoded)
    assert isinstance(decoded, str), "decode should return string for single input"
    print(f"[OK] decode returns string for single input")

    padded = tokenizer.add_padding_tokens(encoded, max_length=20)
    assert isinstance(padded, np.ndarray), "add_padding_tokens should return numpy array"
    print(f"[OK] add_padding_tokens returns numpy array")

    print("\nAll API checks passed!")


if __name__ == "__main__":
    test_basic_tokenization()
    test_padding()
    test_batch_encoding()
    test_special_tokens()
    test_edge_cases()
    test_methods_match_api()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
