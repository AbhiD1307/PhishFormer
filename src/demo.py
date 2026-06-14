import argparse
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import PhishingTransformer
from tokenizer import load_tokenizer, encode


def _plot_attention_bar(ranked: list[tuple], label_str: str,
                        p_phish: float, out_path: Path):
    weights = [w for w, _ in ranked]
    tokens  = [t for _, t in ranked]
    colors  = ["#cb181d" if label_str == "PHISHING" else "#2171b5"] * len(tokens)

    fig, ax = plt.subplots(figsize=(6, max(3, len(tokens) * 0.38)))
    bars = ax.barh(tokens[::-1], weights[::-1], color=colors[::-1],
                   edgecolor="white", linewidth=0.5)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", fontsize=8)
    ax.set_xlabel("CLS Attention Weight (Layer 4, Head 1)", fontsize=9)
    ax.set_title(
        f"Prediction: {label_str}  |  P(phishing) = {p_phish:.4f}",
        fontsize=10,
        color="#cb181d" if label_str == "PHISHING" else "#2171b5",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  Attention bar chart saved → {out_path}")


def predict(checkpoint: str, tokenizer_path: str, text: str,
            max_len: int = 512, top_k: int = 10,
            out_png: str = "results/demo_attention_bar.png"):
    device = torch.device("cpu")
    ckpt   = torch.load(checkpoint, map_location=device, weights_only=False)
    saved  = ckpt.get("args", {})

    model = PhishingTransformer(
        vocab_size=saved.get("vocab_size", 16_000),
        d_model=saved.get("d_model", 256),
        nhead=saved.get("nhead", 8),
        num_layers=saved.get("num_layers", 6),
        d_ff=saved.get("d_ff", 1024),
        dropout=0.0,
        pool=saved.get("pool", "cls_mean"),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tok = load_tokenizer(tokenizer_path)
    ids, mask = encode(tok, text, max_len)
    input_ids    = torch.tensor([ids],  dtype=torch.long)
    padding_mask = torch.tensor([mask], dtype=torch.bool)

    with torch.no_grad():
        logits = model(input_ids, padding_mask)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = logits.argmax(dim=1).item()

    label_str = "PHISHING" if pred == 1 else "LEGITIMATE"
    p_phish   = probs[1].item()
    p_legit   = probs[0].item()

    print(f"\n{'='*50}")
    print(f"  Prediction : {label_str}")
    print(f"  P(phishing): {p_phish:.4f}")
    print(f"  P(legit)   : {p_legit:.4f}")
    print(f"{'='*50}\n")

    # Attention-based token importance (last layer, head 0, [CLS] row)
    attn    = model.get_attention_weights(input_ids, padding_mask)
    n_real  = sum(1 for m in mask if m == 0)
    cls_attn = attn[-1][0, 0, 0, :n_real].numpy()

    token_strs = [tok.id_to_token(i) or str(i) for i in ids[:n_real]]
    ranked = sorted(zip(cls_attn, token_strs), reverse=True)[:top_k]

    print(f"  Top {top_k} tokens by CLS attention (Layer 4, Head 1):")
    for weight, token in ranked:
        bar = "#" * int(weight * 50)
        print(f"    {token:20s}  {weight:.4f}  {bar}")

    _plot_attention_bar(ranked, label_str, p_phish, Path(out_png))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="checkpoints/best_model.pt")
    p.add_argument("--tokenizer",   default="data/tokenizer.json")
    p.add_argument("--text",
                   default="Your PayPal account has been limited! Click below to restore access now.")
    p.add_argument("--max_len",  type=int, default=512)
    p.add_argument("--top_k",    type=int, default=10)
    p.add_argument("--out_png",  default="results/demo_attention_bar.png")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args.checkpoint, args.tokenizer, args.text,
            args.max_len, args.top_k, args.out_png)
