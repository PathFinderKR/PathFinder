import json
import pandas as pd
from typing import Dict, Optional
import torch
from tokenizers import Tokenizer, models, trainers, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from src.config import TokenizerConfig, DatasetConfig


class CharTokenizer:
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        """
        Initialize the character-level tokenizer.

        Args:
            vocab (dict, optional): A pre-defined vocabulary mapping. If None, it will be built from data.
        """

        self.SPECIAL_TOKENS = [
            "<|begin_of_text|>",  # BOS
            "<|end_of_text|>",    # EOS
            "<|UNK|>",            # Unknown token
            "<|PAD|>",            # Padding token
        ]

        if vocab is not None:
            self.char2idx = vocab
            self.idx2char = {idx: char for char, idx in vocab.items()}
            self.vocab_size = len(vocab)
        else:
            self.char2idx: Dict[str, int] = {}
            self.idx2char: Dict[int, str] = {}
            self.vocab_size: int = 0

    def build_vocab(self, text: str):
        """
        Build vocabulary from the provided text.

        Args:
            text (str): The text data to build the vocabulary from.
        """
        unique_chars = sorted(set(text))
        start_idx = len(self.SPECIAL_TOKENS)

        # Character to index mapping
        self.char2idx = {char: idx for idx, char in enumerate(self.SPECIAL_TOKENS)}
        for idx, char in enumerate(unique_chars, start=start_idx):
            if char not in self.char2idx:
                self.char2idx[char] = idx

        # Index to character mapping
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

        self.vocab_size = len(self.char2idx)
        print(f"Vocabulary size: {self.vocab_size}")

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode a string into a tensor of integer token IDs.

        Args:
            text (str): The text to encode.

        Returns:
            torch.Tensor: The encoded tensor.
        """
        ids = []
        for char in text:
            if char in self.char2idx:
                ids.append(self.char2idx[char])
            else:
                ids.append(self.char2idx["<|UNK|>"])
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        """
        Decode a tensor of integer token IDs into a string.

        Args:
            tokens (torch.Tensor): The tensor of token IDs.

        Returns:
            str: The decoded string.
        """
        chars = []
        for idx in tokens:
            if idx in self.idx2char:
                chars.append(self.idx2char[idx])
            else:
                chars.append("?")
        return ''.join(chars)

    def save_vocab(self, file_path: str):
        """
        Save the vocabulary to a JSON file.

        Args:
            file_path (str): The path to save the vocabulary file.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.char2idx, f, ensure_ascii=False, indent=4)
        print(f"Vocabulary saved to {file_path}.")

    def load_vocab(self, file_path: str):
        """
        Load the vocabulary from a JSON file.

        Args:
            file_path (str): The path to the vocabulary file.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            self.char2idx = json.load(f)
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        print(f"Vocabulary loaded from {file_path}.")


def main():
    pass

if __name__ == "__main__":
    main()