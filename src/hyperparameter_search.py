"""
Grid / random hyperparameter search for PhishingTransformer.

Trains a model for each configuration on train split, evaluates on val split,
and writes a ranked results table + bar chart to results/hparam_search/.

Usage:
    # Full grid search (may take hours)
    python src/hyperparameter_search.py

    # Random search — sample 10 configs at random
    python src/hyperparameter_search.py --mode random --n_trials 10

    # Quick smoke-test (2 epochs per trial)
    python src/hyperparameter_search.py --mode random --n_trials 4 --epochs 2
"""

import json
import random
import logging
import argparse
import itertools
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import PhishingTransformer
from augment import AugmentedEmailDataset
from dataset import make_loaders
from train import NoamScheduler
from sklearn.metrics import f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Search space ──────────────────────────────────────────────────────────────
GRID = {
    "d_model":    [128, 256],
    "nhead":      [4, 8],
    "num_layers": [4, 6],
    "d_ff":       [512, 1024],
    "dropout":    [0.1, 0.2],
    "pool":       ["cls", "cls_mean"],
}


def _all_configs() -> list[dict]:
    keys   = list(GRID.keys())
    values = list(GRID.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _random_configs(n: int, seed: int = 42) -> list[dict]:
    rng     = random.Random(seed)
    keys    = list(GRID.keys())
    configs = []
    seen    = set()
    attempts = 0
    while len(configs) < n and attempts < n * 20:
        attempts += 1
        cfg = {k: rng.choice(GRID[k]) for k in keys}
        key = str(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    return configs


def _train_and_eval(cfg: dict, args) -> dict:
    """Train one configuration for `args.epochs` epochs, return val metrics."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    train_ds = AugmentedEmailDataset(
        csv_path       = f"{args.data_dir}/train.csv",
        tokenizer_path = args.tokenizer,
        max_len        = args.max_len,
        augment        = True,
        aug_prob       = 0.40,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)

    _, val_loader, _ = make_loaders(
        data_dir       = args.data_dir,
        tokenizer_path = args.tokenizer,
        max_len        = args.max_len,
        batch_size     = args.batch_size * 2,
        num_workers    = 0,
    )

    # nhead must divide d_model
    if cfg["d_model"] % cfg["nhead"] != 0:
        return None

    model = PhishingTransformer(
        vocab_size  = args.vocab_size,
        d_model     = cfg["d_model"],
        nhead       = cfg["nhead"],
        num_layers  = cfg["num_layers"],
        d_ff        = cfg["d_ff"],
        dropout     = cfg["dropout"],
        pool        = cfg["pool"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98),
                      eps=1e-9, weight_decay=0.01)
    scheduler = NoamScheduler(optimizer, cfg["d_model"],
                              warmup_steps=min(2000, len(train_loader) * 2))

    best_val_f1   = 0.0
    best_val_auc  = 0.0
    best_val_loss = float("inf")

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

        # Validation
        model.eval()
        all_labels, all_preds, all_probs = [], [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ids    = batch["input_ids"].to(device)
                mask   = batch["padding_mask"].to(device)
                labels = batch["label"]
                logits = model(ids, mask)
                val_loss += criterion(logits, labels.to(device)).item()
                probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds  = logits.argmax(1).cpu().numpy()
                all_labels.extend(labels.numpy())
                all_preds.extend(preds)
                all_probs.extend(probs)

        val_loss /= len(val_loader)
        y, yp, ypr = np.array(all_labels), np.array(all_preds), np.array(all_probs)
        f1  = f1_score(y, yp, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y, ypr)
        except Exception:
            auc = 0.0

        if f1 > best_val_f1:
            best_val_f1   = f1
            best_val_auc  = auc
            best_val_loss = val_loss

    n_params = model.num_parameters
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    return {
        **cfg,
        "val_f1":    round(best_val_f1,   4),
        "val_auc":   round(best_val_auc,  4),
        "val_loss":  round(best_val_loss, 4),
        "n_params":  n_params,
    }


def _plot_results(results: list[dict], out_path: Path):
    results = sorted(results, key=lambda r: r["val_f1"], reverse=True)[:15]
    labels  = [f"d{r['d_model']}_h{r['nhead']}_L{r['num_layers']}_ff{r['d_ff']}\n"
               f"drop{r['dropout']}_{r['pool']}" for r in results]
    f1s     = [r["val_f1"]  for r in results]
    aucs    = [r["val_auc"] for r in results]

    x     = np.arange(len(results))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(results) * 0.9), 5))
    b1 = ax.bar(x - width / 2, f1s,  width, label="Val F1",  color="#cb181d", edgecolor="white")
    b2 = ax.bar(x + width / 2, aucs, width, label="Val AUC", color="#2171b5", edgecolor="white")

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5, rotation=0)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(max(0, min(f1s) - 0.05), 1.05)
    ax.set_title("Hyperparameter Search — Top Configurations by Val F1", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved → %s", out_path)


def search(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "grid":
        configs = _all_configs()
        log.info("Grid search: %d configurations", len(configs))
    else:
        configs = _random_configs(args.n_trials, seed=args.seed)
        log.info("Random search: %d configurations (seed=%d)", len(configs), args.seed)

    results = []
    for i, cfg in enumerate(configs, 1):
        log.info("[%d/%d] %s", i, len(configs), cfg)
        r = _train_and_eval(cfg, args)
        if r is None:
            log.warning("  Skipped (nhead does not divide d_model)")
            continue
        log.info("  → val_f1=%.4f  val_auc=%.4f  params=%s",
                 r["val_f1"], r["val_auc"], f"{r['n_params']:,}")
        results.append(r)

    results.sort(key=lambda r: r["val_f1"], reverse=True)

    # Print ranked table
    print(f"\n{'='*72}")
    print(f"  Hyperparameter Search Results  ({args.mode}, {len(results)} configs)")
    print(f"{'='*72}")
    header = f"{'Rank':>4}  {'d_model':>7} {'nhead':>5} {'layers':>6} {'d_ff':>5} "
    header += f"{'drop':>5} {'pool':>8}  {'F1':>6} {'AUC':>6} {'Params':>9}"
    print(header)
    print("-" * 72)
    for rank, r in enumerate(results, 1):
        print(f"  {rank:>2}.  {r['d_model']:>7} {r['nhead']:>5} {r['num_layers']:>6} "
              f"{r['d_ff']:>5} {r['dropout']:>5} {r['pool']:>8}  "
              f"{r['val_f1']:>6.4f} {r['val_auc']:>6.4f} {r['n_params']:>9,}")
    print()

    if results:
        best = results[0]
        print(f"  Best config: d_model={best['d_model']}, nhead={best['nhead']}, "
              f"num_layers={best['num_layers']}, d_ff={best['d_ff']}, "
              f"dropout={best['dropout']}, pool={best['pool']}")
        print(f"  Best val F1: {best['val_f1']:.4f}   val AUC: {best['val_auc']:.4f}")

    with open(out / "hparam_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved → %s/hparam_results.json", out)

    if results:
        _plot_results(results, out / "hparam_search_bar.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       default="random", choices=["grid", "random"])
    p.add_argument("--n_trials",   type=int,   default=16,
                   help="Number of configs to try (random mode only)")
    p.add_argument("--epochs",     type=int,   default=5,
                   help="Training epochs per trial (keep small for speed)")
    p.add_argument("--data_dir",   default="data/processed")
    p.add_argument("--tokenizer",  default="data/tokenizer.json")
    p.add_argument("--vocab_size", type=int,   default=16_000)
    p.add_argument("--max_len",    type=int,   default=256)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--out_dir",    default="results/hparam_search")
    return p.parse_args()


if __name__ == "__main__":
    search(parse_args())
