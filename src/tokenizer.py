import json
import logging
import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

log = logging.getLogger(__name__)

# Special tokens (kept consistent with BERT conventions for drop-in compat)
PAD_TOKEN = "[PAD]"   # id 0
UNK_TOKEN = "[UNK]"   # id 1
CLS_TOKEN = "[CLS]"   # id 2
SEP_TOKEN = "[SEP]"   # id 3
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN]


def train_tokenizer(
    texts: list[str],
    vocab_size: int = 16_000,
    save_path: str = "data/tokenizer.json",
) -> Tokenizer:
    tok = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tok.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)

    # Wrap every sequence with [CLS] ... [SEP]
    tok.post_processor = TemplateProcessing(
        single=f"{CLS_TOKEN} $A {SEP_TOKEN}",
        special_tokens=[
            (CLS_TOKEN, tok.token_to_id(CLS_TOKEN)),
            (SEP_TOKEN, tok.token_to_id(SEP_TOKEN)),
        ],
    )

    tok.save(save_path)
    log.info("Tokenizer saved → %s  (vocab_size=%d)", save_path, tok.get_vocab_size())
    return tok


def load_tokenizer(path: str = "data/tokenizer.json") -> Tokenizer:
    return Tokenizer.from_file(path)


def encode(tokenizer: Tokenizer, text: str, max_len: int = 512) -> tuple[list[int], list[int]]:
    """
    Returns (token_ids, padding_mask) both of length max_len.
    padding_mask: 1 = padding position (ignored by transformer), 0 = real token.
    """
    enc = tokenizer.encode(text)
    ids = enc.ids[:max_len]
    pad_len = max_len - len(ids)
    ids = ids + [0] * pad_len                          # 0 = [PAD]
    mask = [0] * (max_len - pad_len) + [1] * pad_len  # True where padding
    return ids, mask


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/processed/train.csv")
    parser.add_argument("--vocab_size", type=int, default=16_000)
    parser.add_argument("--out", default="data/tokenizer.json")
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)
    texts = df["text"].astype(str).tolist()
    train_tokenizer(texts, vocab_size=args.vocab_size, save_path=args.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
