import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    # nb_metrics.json is a list; grab the test split entry
    if isinstance(data, list):
        for entry in data:
            if entry.get("split") == "test":
                return entry
        return data[-1]
    return data


def _plot_grouped_bar(results: dict, out_path: Path):
    metrics = list(results.keys())
    nb_vals = [results[m]["naive_bayes"]  for m in metrics]
    tr_vals = [results[m]["transformer"]  for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars_nb = ax.bar(x - width / 2, nb_vals, width,
                     label="Naive Bayes", color="#4292c6", edgecolor="white", linewidth=0.6)
    bars_tr = ax.bar(x + width / 2, tr_vals, width,
                     label="Transformer (scratch)", color="#cb181d", edgecolor="white", linewidth=0.6)

    # value labels on top of each bar
    for bar in bars_nb:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_tr:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Model Comparison: Naive Bayes vs Transformer (Test Set)", fontsize=11)
    ax.set_ylim(0, min(1.0, max(tr_vals + nb_vals) + 0.12))
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Bar chart saved       → {out_path}")


def _plot_delta(results: dict, out_path: Path):
    metrics = list(results.keys())
    deltas  = [results[m]["delta"] for m in metrics]
    colors  = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(metrics, deltas, color=colors, edgecolor="white", linewidth=0.6)

    for bar, d in zip(bars, deltas):
        sign = "+" if d >= 0 else ""
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.001 if d >= 0 else -0.003),
                f"{sign}{d:.4f}",
                ha="center", va="bottom" if d >= 0 else "top",
                fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax.set_ylabel("Δ (Transformer − Naive Bayes)", fontsize=10)
    ax.set_title("Performance Gain: Transformer over Naive Bayes", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Delta chart saved     → {out_path}")


def compare(args):
    nb  = load(args.nb_metrics)
    tr  = load(args.transformer_metrics)

    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]

    print(f"\n{'Metric':<14} {'Naive Bayes':>14} {'Transformer':>14} {'Delta':>10}")
    print("-" * 55)
    results = {}
    for m in metrics:
        nb_val = nb.get(m, float("nan"))
        tr_val = tr.get(m, float("nan"))
        delta  = tr_val - nb_val
        sign   = "+" if delta >= 0 else ""
        print(f"{m:<14} {nb_val:>14.4f} {tr_val:>14.4f} {sign}{delta:>9.4f}")
        results[m] = {"naive_bayes": nb_val, "transformer": tr_val, "delta": round(delta, 4)}
    print()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Comparison JSON saved → {out}/comparison.json")

    _plot_grouped_bar(results, out / "comparison_bar.png")
    _plot_delta(results,       out / "comparison_delta.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nb_metrics",           default="results/nb_metrics.json")
    p.add_argument("--transformer_metrics",  default="results/transformer_metrics.json")
    p.add_argument("--out_dir",              default="results")
    return p.parse_args()


if __name__ == "__main__":
    compare(parse_args())
