import argparse
import os
from pathlib import Path

import torch

from cs336_basics.tokenizer import train_bpe


def train(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
):
    """Train BPE and save (vocab, merges, special_tokens) to one file."""
    if len(set(special_tokens)) != len(special_tokens):
        raise ValueError("special_tokens must not contain duplicates")
    if any(token == "" for token in special_tokens):
        raise ValueError("special_tokens must not contain empty strings")
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size must accommodate 256 byte tokens and all special tokens")

    output_path = Path(output_path)
    if output_path.is_dir():
        raise ValueError("output_path must be a file path, for example tokenizers/tinystories.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    torch.save((vocab, merges, list(special_tokens)), output_path)
    print(f"Saved BPE to {output_path}: {len(vocab)} tokens, {len(merges)} merges")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train and save a BPE tokenizer")
    parser.add_argument("--input_path", required=True, help="Path to the training text corpus")
    parser.add_argument("--output_path", required=True, help="Output file, e.g. tokenizers/tinystories.pt")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--special_tokens", nargs="*", default=["<|endoftext|>"])
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    train(args.input_path, args.output_path, args.vocab_size, args.special_tokens)
