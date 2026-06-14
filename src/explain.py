"""
Explainability for the PhishingTransformer.

Two complementary methods:

1. Gradient-based input saliency
   Compute d(loss)/d(embedding) and take the L2 norm per token.
   This highlights which tokens most strongly influence the prediction —
   a model-internals view that does not depend on attention patterns.

2. Attention rollout (Abnar & Zuidema 2020)
   Multiply attention matrices across layers, adding the residual identity
   at each step, to propagate how attention flows from [CLS] to each token.
   More faithful than raw last-layer attention for deep networks.

CLI:
    python explain.py --text "Your PayPal account is limited"
    python explain.py --text "..." --method both --out_dir results/explain
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from model import PhishingTransformer
from tokenizer import load_tokenizer, encode


# Gradient saliency

def gradient_saliency(
    model: PhishingTransformer,
    input_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    target_class: int = 1,
) -> np.ndarray:
    """
    Returns per-token saliency scores (shape: seq_len,) via input-gradient w.r.t. embeddings.
    Tokens with high saliency strongly push the model toward `target_class`.
    """
    model.eval()
    embed = model.embed(input_ids)          # (1, L, D)
    embed.retain_grad()

    # Re-run forward using the embedding directly
    x = model.pos.dropout(embed + model.pos.pe[:, :embed.size(1)])
    x = model.encoder(x, src_key_padding_mask=padding_mask)
    pooled = model._pool(x, padding_mask)
    logits = model.clf(pooled)

    model.zero_grad()
    logits[0, target_class].backward()

    grad = embed.grad[0].detach()           # (L, D)
    saliency = grad.norm(dim=-1).numpy()    # (L,)
    return saliency


# Attention rollout

def attention_rollout(
    attn_weights: list[torch.Tensor],      # list[(1, nhead, L, L)]
    padding_mask: torch.Tensor | None,
    n_real: int,
) -> np.ndarray:
    """
    Compute rollout from [CLS] (position 0) to all other tokens.
    Returns 1-D array of shape (n_real,) — higher = more attended.
    """
    device = attn_weights[0].device
    L      = attn_weights[0].shape[-1]

    rollout = torch.eye(L, device=device).unsqueeze(0)  # (1, L, L)

    for w in attn_weights:                               # (1, nhead, L, L)
        # Average over heads, add residual identity
        w_avg = w.mean(dim=1)                            # (1, L, L)
        w_aug = 0.5 * w_avg + 0.5 * torch.eye(L, device=device)
        # Normalize rows
        w_aug = w_aug / w_aug.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        rollout = torch.bmm(w_aug, rollout)

    cls_rollout = rollout[0, 0, :n_real].cpu().numpy()  # (n_real,)
    return cls_rollout


# Plot helpers

def _bar_chart(
    scores: np.ndarray,
    tokens: list[str],
    title: str,
    xlabel: str,
    color: str,
    out_path: Path,
):
    top_k   = min(15, len(tokens))
    ranked  = sorted(zip(scores, tokens), reverse=True)[:top_k]
    vals    = [v for v, _ in ranked]
    lbls    = [t for _, t in ranked]

    fig, ax = plt.subplots(figsize=(7, max(3, top_k * 0.38)))
    ax.barh(lbls[::-1], vals[::-1], color=color, edgecolor="white", linewidth=0.5)
    for i, v in enumerate(vals[::-1]):
        ax.text(v + max(vals) * 0.01, i, f"{v:.4f}", va="center", fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def _heatmap_all_heads(
    attn_weights: list[torch.Tensor],
    tokens: list[str],
    layer: int,
    out_path: Path,
):
    """Plot all attention heads for a given layer as a grid."""
    w    = attn_weights[layer][0].detach().cpu().numpy()  # (nhead, L, L)
    n, L = w.shape[0], w.shape[1]
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2))
    axes = np.array(axes).flatten()

    for h in range(n):
        sns.heatmap(
            w[h, :L, :L], ax=axes[h],
            xticklabels=tokens, yticklabels=tokens,
            cmap="Blues", vmin=0, vmax=w[h].max(),
            linewidths=0.2, linecolor="white",
            cbar=False,
        )
        axes[h].set_title(f"Head {h+1}", fontsize=9)
        axes[h].tick_params(axis="x", rotation=45, labelsize=6)
        axes[h].tick_params(axis="y", rotation=0,  labelsize=6)

    for h in range(n, len(axes)):
        axes[h].set_visible(False)

    fig.suptitle(f"All Attention Heads — Layer {layer+1}", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# Main

def explain(args):
    device = torch.device("cpu")
    ckpt   = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved  = ckpt.get("args", {})

    model = PhishingTransformer(
        vocab_size  = saved.get("vocab_size", 16_000),
        d_model     = saved.get("d_model",    256),
        nhead       = saved.get("nhead",      8),
        num_layers  = saved.get("num_layers", 6),
        d_ff        = saved.get("d_ff",       1024),
        dropout     = 0.0,
        pool        = saved.get("pool",       "cls_mean"),
    )
    model.load_state_dict(ckpt["model_state"])

    tok = load_tokenizer(args.tokenizer)
    ids, mask = encode(tok, args.text, args.max_len)
    n_real = sum(1 for m in mask if m == 0)

    input_ids    = torch.tensor([ids],  dtype=torch.long)
    padding_mask = torch.tensor([mask], dtype=torch.bool)
    tokens       = [tok.id_to_token(i) or str(i) for i in ids[:n_real]]

    # Forward pass for prediction
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, padding_mask)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = logits.argmax(dim=1).item()

    label   = "PHISHING" if pred == 1 else "LEGITIMATE"
    p_phish = probs[1].item()
    color   = "#cb181d" if pred == 1 else "#2171b5"

    print(f"\n{'='*52}")
    print(f"  Prediction : {label}")
    print(f"  P(phishing): {p_phish:.4f}   P(legit): {probs[0].item():.4f}")
    print(f"{'='*52}\n")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Attention weights (needed for both rollout and heatmap)
    attn = model.get_attention_weights(input_ids, padding_mask)

    if args.method in ("gradient", "both"):
        print("  Computing gradient saliency…")
        sal = gradient_saliency(model, input_ids, padding_mask, target_class=pred)
        sal = sal[:n_real]
        _bar_chart(
            sal, tokens,
            title  = f"Gradient Saliency — {label}  (P={p_phish:.3f})",
            xlabel = "||∂loss/∂embedding||₂",
            color  = color,
            out_path = out / "saliency_bar.png",
        )
        print(f"  Top-5 salient tokens: "
              + ", ".join(t for _, t in sorted(zip(sal, tokens), reverse=True)[:5]))

    if args.method in ("rollout", "both"):
        print("  Computing attention rollout…")
        roll = attention_rollout(attn, padding_mask, n_real)
        _bar_chart(
            roll, tokens,
            title  = f"Attention Rollout — {label}  (P={p_phish:.3f})",
            xlabel = "Rollout weight from [CLS]",
            color  = color,
            out_path = out / "rollout_bar.png",
        )

    # All-heads heatmap for the last layer
    _heatmap_all_heads(attn, tokens, layer=len(attn) - 1,
                       out_path=out / "all_heads_heatmap.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="checkpoints/best_model.pt")
    p.add_argument("--tokenizer",   default="data/tokenizer.json")
    p.add_argument("--text",
                   default="Your PayPal account has been limited! Click below to restore access now.")
    p.add_argument("--max_len",  type=int, default=64)
    p.add_argument("--method",   default="both", choices=["gradient", "rollout", "both"])
    p.add_argument("--out_dir",  default="results/explain")
    return p.parse_args()


if __name__ == "__main__":
    explain(parse_args())
