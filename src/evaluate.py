import json
import logging
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve,
)

from model import PhishingTransformer
from dataset import make_loaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# Plot helpers

def _plot_confusion_matrix(cm: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Reds", ax=ax,
        xticklabels=["Legit", "Phishing"],
        yticklabels=["Legit", "Phishing"],
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title("Transformer — Confusion Matrix (Test)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved → %s", out_path)


def _plot_roc(y: np.ndarray, ypr: np.ndarray, auc: float, out_path: Path):
    fpr, tpr, _ = roc_curve(y, ypr)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot(fpr, tpr, color="#cb181d", lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("Transformer — ROC Curve (Test)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved → %s", out_path)


def _plot_metrics_bar(metrics: dict, out_path: Path):
    keys = list(metrics.keys())
    vals = list(metrics.values())
    colors = ["#41ab5d" if v >= 0.9 else "#cb181d" if v < 0.85 else "#2171b5"
              for v in vals]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(keys, vals, color=colors, edgecolor="white", linewidth=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_xticklabels([k.replace("_", "\n") for k in keys], fontsize=9)
    ax.set_title("Transformer — Test Set Metrics", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved → %s", out_path)


def _plot_prob_dist(y: np.ndarray, ypr: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(ypr[y == 0], bins=50, alpha=0.6, color="#4292c6",
            label="Legit",    density=True)
    ax.hist(ypr[y == 1], bins=50, alpha=0.6, color="#cb181d",
            label="Phishing", density=True)
    ax.set_xlabel("P(phishing)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Transformer — Predicted Probability Distribution (Test)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved → %s", out_path)


# Main

def evaluate(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})

    model = PhishingTransformer(
        vocab_size=saved_args.get("vocab_size", args.vocab_size),
        d_model=saved_args.get("d_model", args.d_model),
        nhead=saved_args.get("nhead", args.nhead),
        num_layers=saved_args.get("num_layers", args.num_layers),
        d_ff=saved_args.get("d_ff", args.d_ff),
        dropout=0.0,
        pool=saved_args.get("pool", "cls_mean"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info("Loaded checkpoint from epoch %d (val_loss=%.4f)",
             ckpt.get("epoch", -1), ckpt.get("val_loss", float("nan")))

    _, _, test_loader = make_loaders(
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer,
        max_len=saved_args.get("max_len", args.max_len),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            ids    = batch["input_ids"].to(device)
            mask   = batch["padding_mask"].to(device)
            labels = batch["label"].cpu().numpy()
            logits = model(ids, mask)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_labels.extend(labels)
            all_preds.extend(preds)
            all_probs.extend(probs)

    y, yp, ypr = np.array(all_labels), np.array(all_preds), np.array(all_probs)

    metrics = {
        "accuracy":  round(float(accuracy_score(y, yp)), 4),
        "f1":        round(float(f1_score(y, yp, average="binary")), 4),
        "precision": round(float(precision_score(y, yp, average="binary")), 4),
        "recall":    round(float(recall_score(y, yp, average="binary")), 4),
        "roc_auc":   round(float(roc_auc_score(y, ypr)), 4),
    }

    print("\n=== TEST SET ===")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v}")
    print("\nClassification report:")
    print(classification_report(y, yp, target_names=["legit", "phishing"]))
    cm = confusion_matrix(y, yp)
    print("Confusion matrix:")
    print(cm)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "transformer_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Saved → %s/transformer_metrics.json", out)

    # ---- PNG outputs ----
    _plot_confusion_matrix(cm,                  out / "transformer_confusion_matrix.png")
    _plot_roc(y, ypr, metrics["roc_auc"],       out / "transformer_roc_curve.png")
    _plot_metrics_bar(metrics,                  out / "transformer_metrics_bar.png")
    _plot_prob_dist(y, ypr,                     out / "transformer_prob_dist.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   default="checkpoints/best_model.pt")
    p.add_argument("--data_dir",     default="data/processed")
    p.add_argument("--tokenizer",    default="data/tokenizer.json")
    p.add_argument("--out_dir",      default="results")
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--max_len",      type=int, default=512)
    p.add_argument("--num_workers",  type=int, default=2)
    p.add_argument("--vocab_size",   type=int, default=16_000)
    p.add_argument("--d_model",      type=int, default=256)
    p.add_argument("--nhead",        type=int, default=4)
    p.add_argument("--num_layers",   type=int, default=4)
    p.add_argument("--d_ff",         type=int, default=512)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
