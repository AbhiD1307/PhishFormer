import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from tokenizer import load_tokenizer, encode


class EmailDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer_path: str, max_len: int = 512):
        self.df = pd.read_csv(csv_path)
        self.tok = load_tokenizer(tokenizer_path)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ids, mask = encode(self.tok, str(row["text"]), self.max_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "padding_mask": torch.tensor(mask, dtype=torch.bool),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def make_loaders(
    data_dir: str = "data/processed",
    tokenizer_path: str = "data/tokenizer.json",
    max_len: int = 512,
    batch_size: int = 32,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    def loader(split: str, shuffle: bool) -> DataLoader:
        ds = EmailDataset(f"{data_dir}/{split}.csv", tokenizer_path, max_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    return loader("train", True), loader("val", False), loader("test", False)
