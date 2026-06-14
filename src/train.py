import json
import math
import logging
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from model import PhishingTransformer
from dataset import make_loaders
from augment import AugmentedEmailDataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# Plot helpers


def _plot_loss_curves(history: list, out_path: Path):
    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(epochs, train_loss, "o-",  color="#2171b5", lw=1.8, label="Train loss")
    ax.plot(epochs, val_loss,   "s--", color="#cb181d", lw=1.8, label="Val loss")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax.set_title("Training & Validation Loss", fontsize=11)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved → %s", out_path)


def _plot_val_acc(history: list, out_path: Path):
    epochs  = [h["epoch"]   for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(epochs, val_acc, "D-", color="#41ab5d", lw=1.8)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title("Validation Accuracy per Epoch", fontsize=11)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_ylim(max(0, min(val_acc) - 0.05), 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved → %s", out_path)


def _plot_lr_schedule(d_model: int, warmup: int, total_steps: int, out_path: Path):
    steps = list(range(1, total_steps + 1))
    lrs   = [(d_model ** -0.5) * min(s ** -0.5, s * warmup ** -1.5) for s in steps]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(steps, lrs, color="#6a51a3", lw=1.5)
    ax.axvline(warmup, color="gray", linestyle=":", lw=1.2, label=f"Warmup ({warmup} steps)")
    ax.set_xlabel("Training Step", fontsize=10)
    ax.set_ylabel("Learning Rate", fontsize=10)
    ax.set_title("Noam Learning-Rate Schedule", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved → %s", out_path)


# Learning-rate schedule: Noam / "Attention is All You Need"

class NoamScheduler:
    """lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})"""

    def __init__(self, optimizer: AdamW, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        s = self.step_num
        lr = (self.d_model ** -0.5) * min(s ** -0.5, s * self.warmup ** -1.5)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# Training loop

def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    log.info("Device: %s", device)

    # Mixed precision only on CUDA
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # Validation and test use the plain loader (no augmentation)
    _, val_loader, _ = make_loaders(
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Training uses AugmentedEmailDataset for on-the-fly data augmentation
    train_ds = AugmentedEmailDataset(
        csv_path=f"{args.data_dir}/train.csv",
        tokenizer_path=args.tokenizer,
        max_len=args.max_len,
        augment=args.augment,
        aug_prob=args.aug_prob,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
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

    log.info("Model parameters: %s  pool=%s", f"{model.num_parameters:,}", args.pool)

    # Label smoothing reduces overconfidence on noisy email data
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # AdamW decouples weight decay from the adaptive LR update — prevents
    # embedding weights from being over-regularized (unlike L2 in Adam).
    optimizer = AdamW(
        model.parameters(),
        lr=1.0,  # Noam scheduler sets the actual LR
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=args.weight_decay,
    )
    scheduler = NoamScheduler(optimizer, args.d_model, args.warmup_steps)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    accum_steps = max(1, args.grad_accum)

    for epoch in range(1, args.epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            ids    = batch["input_ids"].to(device)
            mask   = batch["padding_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(ids, mask)
                loss   = criterion(logits, labels) / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            train_loss += loss.item() * accum_steps

        train_loss /= len(train_loader)

        # -- Validate --
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                ids    = batch["input_ids"].to(device)
                mask   = batch["padding_mask"].to(device)
                labels = batch["label"].to(device)
                logits = model(ids, mask)
                val_loss += criterion(logits, labels).item()
                preds    = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_loss /= len(val_loader)
        val_acc   = correct / total
        lr        = optimizer.param_groups[0]["lr"]

        log.info("Epoch %2d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | lr=%.2e",
                 epoch, train_loss, val_loss, val_acc, lr)

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_acc": val_acc, "lr": lr,
        })

        # -- Checkpoint --
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }, ckpt_dir / "best_model.pt")
            log.info("  → saved best checkpoint (val_loss=%.4f)", val_loss)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info("Early stopping at epoch %d", epoch)
                break

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    log.info("Training complete. History saved to %s/history.json", ckpt_dir)

    _plot_loss_curves(history, ckpt_dir / "loss_curves.png")
    _plot_val_acc(history,     ckpt_dir / "val_acc.png")
    _plot_lr_schedule(args.d_model, args.warmup_steps,
                      len(history) * len(train_loader),
                      ckpt_dir / "lr_schedule.png")


# CLI

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    # Model
    p.add_argument("--vocab_size",  type=int,   default=16_000)
    p.add_argument("--d_model",     type=int,   default=256)
    p.add_argument("--nhead",       type=int,   default=8)
    p.add_argument("--num_layers",  type=int,   default=6)
    p.add_argument("--d_ff",        type=int,   default=1024)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--pool",        default="cls_mean",
                   choices=["cls", "mean", "cls_mean"])
    # Training
    p.add_argument("--epochs",           type=int,   default=15)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--max_len",          type=int,   default=512)
    p.add_argument("--warmup_steps",     type=int,   default=4_000)
    p.add_argument("--patience",         type=int,   default=4)
    p.add_argument("--num_workers",      type=int,   default=2)
    p.add_argument("--weight_decay",     type=float, default=0.01)
    p.add_argument("--label_smoothing",  type=float, default=0.05)
    p.add_argument("--grad_accum",       type=int,   default=1,
                   help="Gradient accumulation steps (effective_batch = batch_size * grad_accum)")
    p.add_argument("--augment",          action="store_true", default=True,
                   help="Apply on-the-fly text augmentation during training")
    p.add_argument("--no_augment",       dest="augment", action="store_false",
                   help="Disable text augmentation")
    p.add_argument("--aug_prob",         type=float, default=0.50,
                   help="Probability of augmenting each training sample")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
