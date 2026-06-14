import json
import logging
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from model import PhishingTransformer
from tokenizer import load_tokenizer, encode

log = logging.getLogger(__name__)


# 1. Loss curves


def plot_loss_curves(history_path: str, out_path: str = "results/loss_curves.png"):
    with open(history_path) as f:
        history = json.load(f)

    epochs      = [h["epoch"]      for h in history]
    train_loss  = [h["train_loss"] for h in history]
    val_loss    = [h["val_loss"]   for h in history]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(epochs, train_loss, "o-", color="#2171b5", label="Train loss", linewidth=1.8)
    ax.plot(epochs, val_loss,   "s--", color="#cb181d", label="Val loss",   linewidth=1.8)

    # Optional NB baseline cross-entropy reference
    ax.axhline(y=0.41, color="gray", linestyle=":", linewidth=1.2, label="NB baseline")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax.set_title("Training & Validation Loss", fontsize=12)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Loss curve saved → %s", out_path)


# 2. Attention heatmap


def get_tokens(tokenizer, text: str, max_len: int = 32) -> list[str]:
    """Return decoded token strings (up to max_len) for axis labels."""
    enc = tokenizer.encode(text)
    ids = enc.ids[:max_len]
    return [tokenizer.id_to_token(i) or str(i) for i in ids]


def plot_attention_heatmap(
    checkpoint: str,
    text: str,
    layer: int = 3,      # 0-indexed (layer 4 = index 3)
    head: int = 0,
    max_len: int = 32,
    out_path: str = "results/attention_heatmap.png",
    # Model dims – override if non-default
    vocab_size: int = 16_000,
    d_model: int = 256,
    nhead: int = 4,
    num_layers: int = 4,
    d_ff: int = 512,
    tokenizer_path: str = "data/tokenizer.json",
):
    device = torch.device("cpu")

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})

    model = PhishingTransformer(
        vocab_size=saved.get("vocab_size", vocab_size),
        d_model=saved.get("d_model", d_model),
        nhead=saved.get("nhead", nhead),
        num_layers=saved.get("num_layers", num_layers),
        d_ff=saved.get("d_ff", d_ff),
        dropout=0.0,
        pool=saved.get("pool", "cls_mean"),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tok = load_tokenizer(tokenizer_path)
    ids, mask = encode(tok, text, max_len)
    input_ids   = torch.tensor([ids],  dtype=torch.long)
    padding_mask = torch.tensor([mask], dtype=torch.bool)

    token_strs = get_tokens(tok, text, max_len)
    n_real = sum(1 for m in mask if m == 0)  # non-padding positions
    token_strs = token_strs[:n_real]

    attn_weights = model.get_attention_weights(input_ids, padding_mask)
    # attn_weights[layer]: (1, nhead, seq_len, seq_len)
    w = attn_weights[layer][0, head, :n_real, :n_real].detach().numpy()

    fig, ax = plt.subplots(figsize=(max(5, n_real * 0.55), max(4, n_real * 0.5)))
    sns.heatmap(
        w, ax=ax,
        xticklabels=token_strs, yticklabels=token_strs,
        cmap="Blues", vmin=0, vmax=w.max(),
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Attention weight"},
        annot=n_real <= 16,
        fmt=".2f" if n_real <= 16 else "",
        annot_kws={"size": 7},
    )
    ax.set_title(f"Self-Attention Heatmap — Layer {layer+1}, Head {head+1}", fontsize=11)
    ax.set_xlabel("Key token", fontsize=9)
    ax.set_ylabel("Query token", fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Attention heatmap saved → %s", out_path)


# CLI


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")

    lc = sub.add_parser("loss", help="Plot loss curves")
    lc.add_argument("--history",  default="checkpoints/history.json")
    lc.add_argument("--out",      default="results/loss_curves.png")

    ah = sub.add_parser("attn", help="Plot attention heatmap")
    ah.add_argument("--checkpoint",     default="checkpoints/best_model.pt")
    ah.add_argument("--tokenizer",      default="data/tokenizer.json")
    ah.add_argument("--text",
                    default="Your PayPal account has been limited click below to verify now")
    ah.add_argument("--layer",   type=int, default=3)
    ah.add_argument("--head",    type=int, default=0)
    ah.add_argument("--max_len", type=int, default=32)
    ah.add_argument("--out",     default="results/attention_heatmap.png")

    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.command == "loss":
        plot_loss_curves(args.history, args.out)
    elif args.command == "attn":
        plot_attention_heatmap(
            checkpoint=args.checkpoint,
            text=args.text,
            layer=args.layer,
            head=args.head,
            max_len=args.max_len,
            out_path=args.out,
            tokenizer_path=args.tokenizer,
        )
    else:
        print("Usage: python visualize.py {loss|attn} --help")
