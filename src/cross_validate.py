"""
5-fold stratified cross-validation for the PhishingTransformer and Naive Bayes.

Combines train + val CSVs, runs K-fold stratified splits, trains each model
from scratch per fold, and reports mean ± std across folds.

Usage:
    cd Project && python src/cross_validate.py
    python src/cross_validate.py --folds 5 --epochs 8 --out_dir results/cv
"""

import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

from model import PhishingTransformer
from dataset import EmailDataset
from train import NoamScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _metric_dict(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, average="binary", zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="binary", zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, average="binary", zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def _summarize(fold_results: list[dict]) -> dict:
    keys = list(fold_results[0].keys())
    return {
        k: {
            "mean": round(float(np.mean([r[k] for r in fold_results])), 4),
            "std":  round(float(np.std( [r[k] for r in fold_results])), 4),
        }
        for k in keys
    }


def _train_fold_transformer(
    train_idx, val_idx, full_ds, args, device
) -> dict:
    train_loader = DataLoader(
        Subset(full_ds, train_idx),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        Subset(full_ds, val_idx),
        batch_size=args.batch_size * 2, shuffle=False, num_workers=0,
    )

    model = PhishingTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pool=args.pool,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98),
                      eps=1e-9, weight_decay=0.01)
    scheduler = NoamScheduler(optimizer, args.d_model, warmup_steps=min(1000, len(train_loader) * 2))

    best_val_loss = float("inf")
    best_state    = None
    patience      = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            ids    = batch["input_ids"].to(device)
            mask   = batch["padding_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, mask), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ids    = batch["input_ids"].to(device)
                mask   = batch["padding_mask"].to(device)
                labels = batch["label"].to(device)
                val_loss += criterion(model(ids, mask), labels).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience      = 0
        else:
            patience += 1
            if patience >= 3:
                break

    model.load_state_dict(best_state)
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            ids    = batch["input_ids"].to(device)
            mask   = batch["padding_mask"].to(device)
            labels = batch["label"].numpy()
            logits = model(ids, mask)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_labels.extend(labels)
            all_preds.extend(preds)
            all_probs.extend(probs)

    return _metric_dict(np.array(all_labels), np.array(all_preds), np.array(all_probs))


def cross_validate(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    log.info("Device: %s", device)

    # Combine train + val for CV (keep test held out)
    train_df = pd.read_csv(f"{args.data_dir}/train.csv")
    val_df   = pd.read_csv(f"{args.data_dir}/val.csv")
    df       = pd.concat([train_df, val_df], ignore_index=True)
    labels   = df["label"].values

    log.info("CV dataset: %d samples  (train+val combined)", len(df))

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    # ---- Transformer CV ----
    log.info("Running %d-fold CV on Transformer…", args.folds)
    full_ds = EmailDataset(
        csv_path       = None,  # we pass df directly below
        tokenizer_path = args.tokenizer,
        max_len        = args.max_len,
    )
    full_ds.df = df.reset_index(drop=True)  # override internal df

    tr_fold_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels), 1):
        log.info("  Fold %d/%d …", fold, args.folds)
        metrics = _train_fold_transformer(train_idx, val_idx, full_ds, args, device)
        log.info("  Fold %d: %s", fold, metrics)
        tr_fold_results.append(metrics)

    tr_summary = _summarize(tr_fold_results)

    # ---- Naive Bayes CV ----
    log.info("Running %d-fold CV on Naive Bayes…", args.folds)
    nb_fold_results = []
    texts = df["text"].astype(str).values

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels), 1):
        nb = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2),
                                      sublinear_tf=True, min_df=2)),
            ("clf",   MultinomialNB(alpha=0.1)),
        ])
        nb.fit(texts[train_idx], labels[train_idx])
        y_pred = nb.predict(texts[val_idx])
        y_prob = nb.predict_proba(texts[val_idx])[:, 1]
        metrics = _metric_dict(labels[val_idx], y_pred, y_prob)
        log.info("  NB Fold %d: %s", fold, metrics)
        nb_fold_results.append(metrics)

    nb_summary = _summarize(nb_fold_results)

    # ---- Print summary ----
    print(f"\n{'='*60}")
    print(f"  {args.folds}-Fold Cross-Validation Results")
    print(f"{'='*60}")
    print(f"\n{'Metric':<14} {'Transformer (mean±std)':>26} {'Naive Bayes (mean±std)':>26}")
    print("-" * 68)
    for m in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
        tr = tr_summary[m]
        nb = nb_summary[m]
        print(f"{m:<14} {tr['mean']:>10.4f} ± {tr['std']:.4f}        "
              f"{nb['mean']:>10.4f} ± {nb['std']:.4f}")
    print()

    results = {
        "transformer": {"per_fold": tr_fold_results, "summary": tr_summary},
        "naive_bayes": {"per_fold": nb_fold_results, "summary": nb_summary},
    }
    with open(out / "cv_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved → %s/cv_results.json", out)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="data/processed")
    p.add_argument("--tokenizer",   default="data/tokenizer.json")
    p.add_argument("--out_dir",     default="results/cv")
    p.add_argument("--folds",       type=int,   default=5)
    p.add_argument("--epochs",      type=int,   default=8)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--max_len",     type=int,   default=256)
    p.add_argument("--vocab_size",  type=int,   default=16_000)
    p.add_argument("--d_model",     type=int,   default=256)
    p.add_argument("--nhead",       type=int,   default=8)
    p.add_argument("--num_layers",  type=int,   default=6)
    p.add_argument("--d_ff",        type=int,   default=1024)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--pool",        default="cls_mean")
    return p.parse_args()


if __name__ == "__main__":
    cross_validate(parse_args())
