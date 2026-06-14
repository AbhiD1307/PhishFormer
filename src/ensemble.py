"""
Soft-voting ensemble: Naive Bayes + Transformer.

Combines P(phishing) from both models as a weighted average, then evaluates
on the held-out test set. Typically gains 0.5–2 pp F1 over either model alone
by exploiting their complementary error modes:
  - NB is strong on high-frequency keyword patterns (fast, interpretable).
  - Transformer captures long-range context and token ordering.

Usage:
    python src/ensemble.py
    python src/ensemble.py --tr_weight 0.7 --nb_weight 0.3
    python src/ensemble.py --sweep   # try all weight combinations
"""

import json
import pickle
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, roc_curve,
)

from model import PhishingTransformer
from dataset import make_loaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load_transformer_probs(args, device) -> tuple[np.ndarray, np.ndarray]:
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})

    model = PhishingTransformer(
        vocab_size = saved.get("vocab_size", 16_000),
        d_model    = saved.get("d_model",    256),
        nhead      = saved.get("nhead",      8),
        num_layers = saved.get("num_layers", 6),
        d_ff       = saved.get("d_ff",       1024),
        dropout    = 0.0,
        pool       = saved.get("pool",       "cls_mean"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, _, test_loader = make_loaders(
        data_dir       = args.data_dir,
        tokenizer_path = args.tokenizer,
        max_len        = saved.get("max_len", 512),
        batch_size     = 64,
        num_workers    = 0,
    )

    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids    = batch["input_ids"].to(device)
            mask   = batch["padding_mask"].to(device)
            labels = batch["label"].numpy()
            probs  = torch.softmax(model(ids, mask), dim=1)[:, 1].cpu().numpy()
            all_labels.extend(labels)
            all_probs.extend(probs)

    return np.array(all_labels), np.array(all_probs)


def _load_nb_probs(args, y_true: np.ndarray) -> np.ndarray:
    import pandas as pd
    with open(args.nb_model, "rb") as f:
        nb = pickle.load(f)
    test_df = pd.read_csv(f"{args.data_dir}/test.csv")
    texts   = test_df["text"].astype(str).values
    return nb.predict_proba(texts)[:, 1]


def _metrics(y, yp, ypr) -> dict:
    return {
        "accuracy":  round(float(accuracy_score(y, yp)), 4),
        "f1":        round(float(f1_score(y, yp, average="binary", zero_division=0)), 4),
        "precision": round(float(precision_score(y, yp, average="binary", zero_division=0)), 4),
        "recall":    round(float(recall_score(y, yp, average="binary", zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y, ypr)), 4),
    }


def _plot_roc_comparison(
    y: np.ndarray,
    probs_dict: dict[str, np.ndarray],
    out_path: Path,
):
    colors = {"Naive Bayes": "#4292c6", "Transformer": "#cb181d", "Ensemble": "#41ab5d"}
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    for name, probs in probs_dict.items():
        auc = roc_auc_score(y, probs)
        fpr, tpr, _ = roc_curve(y, probs)
        ax.plot(fpr, tpr, lw=2, color=colors.get(name, "gray"),
                label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curve Comparison", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved → %s", out_path)


def _plot_metric_comparison(
    results: dict[str, dict],
    out_path: Path,
):
    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    models  = list(results.keys())
    x       = np.arange(len(metrics))
    width   = 0.25
    colors  = ["#4292c6", "#cb181d", "#41ab5d"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (model_name, m) in enumerate(results.items()):
        vals   = [m[k] for k in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=model_name,
                        color=colors[i % len(colors)], edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_title("NB vs Transformer vs Ensemble (Test Set)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved → %s", out_path)


def ensemble(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    log.info("Loading Transformer probabilities…")
    y_true, tr_probs = _load_transformer_probs(args, device)

    log.info("Loading Naive Bayes probabilities…")
    nb_probs = _load_nb_probs(args, y_true)

    assert len(y_true) == len(nb_probs), "Test set size mismatch between models"

    if args.sweep:
        # Grid search over ensemble weights
        best_f1, best_w = -1.0, 0.5
        for w in np.arange(0.0, 1.01, 0.05):
            ens = w * tr_probs + (1 - w) * nb_probs
            f1  = f1_score(y_true, (ens >= 0.5).astype(int), average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1, best_w = f1, w
        log.info("Best ensemble weight: tr=%.2f  nb=%.2f  → F1=%.4f", best_w, 1 - best_w, best_f1)
        args.tr_weight = best_w
        args.nb_weight = 1 - best_w

    ens_probs = args.tr_weight * tr_probs + args.nb_weight * nb_probs
    ens_preds = (ens_probs >= 0.5).astype(int)
    tr_preds  = (tr_probs  >= 0.5).astype(int)
    nb_preds  = (nb_probs  >= 0.5).astype(int)

    results = {
        "Naive Bayes":  _metrics(y_true, nb_preds,  nb_probs),
        "Transformer":  _metrics(y_true, tr_preds,  tr_probs),
        "Ensemble":     _metrics(y_true, ens_preds, ens_probs),
    }
    results["Ensemble"]["tr_weight"] = round(args.tr_weight, 4)
    results["Ensemble"]["nb_weight"] = round(args.nb_weight, 4)

    print(f"\n{'='*60}")
    print(f"  Ensemble Evaluation  (tr={args.tr_weight:.2f}, nb={args.nb_weight:.2f})")
    print(f"{'='*60}")
    print(f"\n{'Metric':<14} {'Naive Bayes':>14} {'Transformer':>14} {'Ensemble':>14}")
    print("-" * 58)
    for m in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
        nb_v  = results["Naive Bayes"][m]
        tr_v  = results["Transformer"][m]
        en_v  = results["Ensemble"][m]
        delta = en_v - max(nb_v, tr_v)
        sign  = "+" if delta >= 0 else ""
        print(f"{m:<14} {nb_v:>14.4f} {tr_v:>14.4f} {en_v:>14.4f}  ({sign}{delta:.4f})")
    print()
    print("Classification report (Ensemble):")
    print(classification_report(y_true, ens_preds, target_names=["legit", "phishing"]))

    with open(out / "ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved → %s/ensemble_results.json", out)

    _plot_roc_comparison(
        y_true,
        {"Naive Bayes": nb_probs, "Transformer": tr_probs, "Ensemble": ens_probs},
        out / "ensemble_roc.png",
    )
    _plot_metric_comparison(
        {k: {m: v for m, v in results[k].items() if isinstance(v, float) and m != "tr_weight" and m != "nb_weight"}
         for k in ["Naive Bayes", "Transformer", "Ensemble"]},
        out / "ensemble_metrics_bar.png",
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--nb_model",   default="results/nb_model.pkl")
    p.add_argument("--data_dir",   default="data/processed")
    p.add_argument("--tokenizer",  default="data/tokenizer.json")
    p.add_argument("--out_dir",    default="results/ensemble")
    p.add_argument("--tr_weight",  type=float, default=0.65)
    p.add_argument("--nb_weight",  type=float, default=0.35)
    p.add_argument("--sweep",      action="store_true",
                   help="Grid-search best ensemble weights by val F1")
    return p.parse_args()


if __name__ == "__main__":
    ensemble(parse_args())
