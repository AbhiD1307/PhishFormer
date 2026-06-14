import json
import logging
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# Plot helpers


def _plot_confusion_matrix(cm: np.ndarray, split: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Legit", "Phishing"],
        yticklabels=["Legit", "Phishing"],
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(f"Naive Bayes — Confusion Matrix ({split})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved → %s", out_path)


def _plot_roc(y_true: np.ndarray, y_prob: np.ndarray,
              auc: float, split: str, out_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot(fpr, tpr, color="#2171b5", lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(f"Naive Bayes — ROC Curve ({split})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved → %s", out_path)


def _plot_metrics_bar(all_results: list[dict], out_path: Path):
    metric_keys = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    splits = [r["split"] for r in all_results]
    x = np.arange(len(metric_keys))
    width = 0.35
    colors = ["#4292c6", "#cb181d", "#41ab5d"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (res, color) in enumerate(zip(all_results, colors)):
        vals = [res[m] for m in metric_keys]
        offset = (i - len(all_results) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=res["split"].capitalize(),
                      color=color, edgecolor="white", linewidth=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metric_keys], fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_title("Naive Bayes — Metrics by Split", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved → %s", out_path)


# Main


def train_nb(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(f"{args.data_dir}/train.csv")
    val_df   = pd.read_csv(f"{args.data_dir}/val.csv")
    test_df  = pd.read_csv(f"{args.data_dir}/test.csv")

    X_train, y_train = train_df["text"].astype(str), train_df["label"]
    X_val,   y_val   = val_df["text"].astype(str),   val_df["label"]
    X_test,  y_test  = test_df["text"].astype(str),  test_df["label"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            min_df=2,
        )),
        ("clf", MultinomialNB(alpha=0.1)),
    ])

    log.info("Training Naive Bayes…")
    model.fit(X_train, y_train)

    stored_probs = {}

    def evaluate(X, y, split_name):
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        stored_probs[split_name] = (np.array(y), y_prob)
        metrics = {
            "split":     split_name,
            "accuracy":  round(accuracy_score(y, y_pred), 4),
            "f1":        round(f1_score(y, y_pred, average="binary"), 4),
            "precision": round(precision_score(y, y_pred, average="binary"), 4),
            "recall":    round(recall_score(y, y_pred, average="binary"), 4),
            "roc_auc":   round(roc_auc_score(y, y_prob), 4),
        }
        log.info("[%s] acc=%.4f  f1=%.4f  prec=%.4f  rec=%.4f  roc-auc=%.4f",
                 split_name, metrics["accuracy"], metrics["f1"],
                 metrics["precision"], metrics["recall"], metrics["roc_auc"])
        print(f"\n=== {split_name.upper()} ===")
        print(classification_report(y, y_pred, target_names=["legit", "phishing"]))
        cm = confusion_matrix(y, y_pred)
        print("Confusion matrix:\n", cm)
        return metrics, y_pred, cm

    val_metrics,  val_pred,  _       = evaluate(X_val,  y_val,  "val")
    test_metrics, test_pred, test_cm = evaluate(X_test, y_test, "test")
    results = [val_metrics, test_metrics]

    # ---- PNG outputs ----
    _plot_confusion_matrix(test_cm, "test", out / "nb_confusion_matrix.png")

    y_test_arr, test_prob = stored_probs["test"]
    _plot_roc(y_test_arr, test_prob, test_metrics["roc_auc"], "test",
              out / "nb_roc_curve.png")

    _plot_metrics_bar(results, out / "nb_metrics_bar.png")

    # ---- JSON + model ----
    with open(out / "nb_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("Metrics saved → %s/nb_metrics.json", out)

    with open(out / "nb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    log.info("Model saved → %s/nb_model.pkl", out)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--out_dir",  default="results")
    return p.parse_args()


if __name__ == "__main__":
    train_nb(parse_args())
