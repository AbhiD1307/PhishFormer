"""
Email-specific text augmentation for training data diversity.

Three lightweight strategies that don't require external resources:
  1. random_deletion  — randomly drop words with probability p
  2. random_swap      — randomly swap two words n times
  3. truncate_body    — keep only the first k% of words (simulates short previews)

AugmentedEmailDataset wraps the base EmailDataset and applies one randomly
chosen augmentation to each training sample on-the-fly, doubling effective
data diversity without storing extra copies on disk.
"""

import random
from typing import Callable

import torch
import pandas as pd
from torch.utils.data import Dataset

from tokenizer import load_tokenizer, encode


def random_deletion(text: str, p: float = 0.10) -> str:
    words = text.split()
    if len(words) <= 1:
        return text
    kept = [w for w in words if random.random() > p]
    return " ".join(kept) if kept else words[0]


def random_swap(text: str, n: int = 2) -> str:
    words = text.split()
    if len(words) < 2:
        return text
    words = words[:]
    for _ in range(n):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)


def truncate_body(text: str, keep_frac: float = 0.75) -> str:
    words = text.split()
    cut = max(1, int(len(words) * keep_frac))
    return " ".join(words[:cut])


_AUGMENTATIONS: list[Callable[[str], str]] = [
    lambda t: random_deletion(t, p=0.10),
    lambda t: random_swap(t, n=2),
    lambda t: truncate_body(t, keep_frac=0.80),
]


class AugmentedEmailDataset(Dataset):
    """
    Wraps a CSV split and applies random augmentation during training.

    Each __getitem__ call independently samples one augmentation, so the
    same email appears differently across epochs — cheap curriculum effect.

    Args:
        csv_path: path to train/val/test CSV with 'text' and 'label' columns.
        tokenizer_path: BPE tokenizer JSON.
        max_len: sequence length cap.
        augment: whether to apply augmentation (disable for val/test).
        aug_prob: probability of augmenting a given sample (default 0.5).
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer_path: str,
        max_len: int = 512,
        augment: bool = True,
        aug_prob: float = 0.50,
    ):
        self.df         = pd.read_csv(csv_path)
        self.tok        = load_tokenizer(tokenizer_path)
        self.max_len    = max_len
        self.augment    = augment
        self.aug_prob   = aug_prob

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row  = self.df.iloc[idx]
        text = str(row["text"])

        if self.augment and random.random() < self.aug_prob:
            fn   = random.choice(_AUGMENTATIONS)
            text = fn(text)

        ids, mask = encode(self.tok, text, self.max_len)
        return {
            "input_ids":    torch.tensor(ids,  dtype=torch.long),
            "padding_mask": torch.tensor(mask, dtype=torch.bool),
            "label":        torch.tensor(int(row["label"]), dtype=torch.long),
        }
