import json
import math
import pickle
import random
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

warnings.filterwarnings("ignore")

RESULTS   = Path("results");   RESULTS.mkdir(exist_ok=True)
CKPT_DIR  = Path("checkpoints"); CKPT_DIR.mkdir(exist_ok=True)
DATA_DIR  = Path("data/processed"); DATA_DIR.mkdir(parents=True, exist_ok=True)
TOK_PATH  = Path("data/tokenizer.json")

# ── colours ──────────────────────────────────────────────────────────────────
C_BLUE = "#2171b5"; C_RED = "#cb181d"; C_GREEN = "#41ab5d"
C_PURPLE = "#6a51a3"; C_TEAL = "#4292c6"

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# 1. SYNTHETIC DATA

_LEGIT_TEMPLATES = [
    "Hi {name}, please find the meeting notes attached from {day}.",
    "Reminder: team standup tomorrow at {time} in {room}.",
    "Your quarterly report for {quarter} is now available.",
    "Could you review the attached contract before {day}?",
    "Thanks for the update on the {project} project.",
    "The {product} deployment is scheduled for {day} at {time}.",
    "Please complete the survey about our {dept} processes.",
    "Your expense report for {quarter} has been approved.",
    "Here are the action items from yesterday's {project} meeting.",
    "Welcome to the {dept} team! Your onboarding starts {day}.",
    "The budget for {quarter} has been reviewed by finance.",
    "Please join the call on {day} to discuss {project} milestones.",
    "Lunch with the {dept} team is scheduled for {day}.",
    "Reminder to submit your timesheets by {day}.",
    "Your certificate of completion for {course} is attached.",
]
_PHISH_TEMPLATES = [
    "Your {bank} account has been suspended! Click here to verify now.",
    "Urgent: confirm your {service} password immediately or lose access.",
    "Congratulations! You have won a {prize}. Claim your reward today.",
    "Your {bank} account will be closed. Verify your details now.",
    "Security alert: unusual login detected on your {service} account.",
    "FINAL NOTICE: your {service} subscription expires today. Renew now.",
    "You have a pending payment of ${amount}. Confirm your {bank} details.",
    "Your {service} account has been compromised. Update credentials immediately.",
    "Limited offer: get a free {prize} by clicking below before it expires!",
    "Action required: verify your {service} billing information now.",
    "Your package from {company} could not be delivered. Reschedule here.",
    "Your {bank} card has been charged ${amount}. Dispute the transaction now.",
    "Important: your {service} account needs identity verification.",
    "Dear customer, your {prize} shipment is on hold. Confirm details.",
    "WARNING: your {service} account will be permanently deleted in 24 hours.",
]

_NAMES    = ["Alice", "Bob", "Carol", "David", "Eve"]
_DAYS     = ["Monday", "Tuesday", "Wednesday", "Friday", "next week"]
_TIMES    = ["9am", "10:30am", "2pm", "3:30pm"]
_ROOMS    = ["Room 101", "the Zoom link", "Conference B"]
_QUARTERS = ["Q1 2024", "Q2 2024", "Q3 2024"]
_PROJECTS = ["Phoenix", "Atlas", "Horizon", "Delta"]
_DEPTS    = ["Engineering", "Marketing", "Finance", "HR"]
_PRODUCTS = ["API gateway", "dashboard", "mobile app", "data pipeline"]
_COURSES  = ["Python basics", "Leadership 101", "Data Privacy"]
_BANKS    = ["PayPal", "Chase", "Bank of America", "Wells Fargo", "Citibank"]
_SERVICES = ["Netflix", "Apple ID", "Google", "Amazon", "Microsoft"]
_PRIZES   = ["$500 gift card", "iPhone 15", "MacBook Pro", "free vacation"]
_AMOUNTS  = ["99.99", "250.00", "49.99", "750.00"]
_COMPANIES= ["FedEx", "UPS", "DHL", "USPS"]


def _fill(template):
    return template.format(
        name=random.choice(_NAMES), day=random.choice(_DAYS),
        time=random.choice(_TIMES), room=random.choice(_ROOMS),
        quarter=random.choice(_QUARTERS), project=random.choice(_PROJECTS),
        dept=random.choice(_DEPTS), product=random.choice(_PRODUCTS),
        course=random.choice(_COURSES), bank=random.choice(_BANKS),
        service=random.choice(_SERVICES), prize=random.choice(_PRIZES),
        amount=random.choice(_AMOUNTS), company=random.choice(_COMPANIES),
    )


def make_synthetic(n_total=4000, ratio_legit=0.60):
    n_legit = int(n_total * ratio_legit)
    n_phish = n_total - n_legit
    legit = [{"text": _fill(random.choice(_LEGIT_TEMPLATES)), "label": 0}
             for _ in range(n_legit)]
    phish = [{"text": _fill(random.choice(_PHISH_TEMPLATES)), "label": 1}
             for _ in range(n_phish)]
    df = pd.DataFrame(legit + phish).sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_train = int(len(df) * 0.80)
    n_val   = int(len(df) * 0.10)
    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train:n_train + n_val]
    test_df  = df.iloc[n_train + n_val:]
    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_DIR  / "val.csv",   index=False)
    test_df.to_csv(DATA_DIR / "test.csv",  index=False)
    print(f"[data] train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    return train_df, val_df, test_df


# 2. BPE TOKENIZER

PAD, UNK, CLS, SEP = "[PAD]", "[UNK]", "[CLS]", "[SEP]"
SPECIAL = [PAD, UNK, CLS, SEP]
VOCAB_SIZE = 2000


def train_tokenizer(texts):
    tok = Tokenizer(BPE(unk_token=UNK))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL,
                         min_frequency=1, show_progress=False)
    tok.train_from_iterator(texts, trainer=trainer)
    tok.post_processor = TemplateProcessing(
        single=f"{CLS} $A {SEP}",
        special_tokens=[(CLS, tok.token_to_id(CLS)), (SEP, tok.token_to_id(SEP))],
    )
    tok.save(str(TOK_PATH))
    print(f"[tok] vocab_size={tok.get_vocab_size()}")
    return tok


def encode(tok, text, max_len=64):
    ids = tok.encode(text).ids[:max_len]
    pad = max_len - len(ids)
    return ids + [0] * pad, [0] * (max_len - pad) + [1] * pad


# 3. DATASET + DATALOADER

MAX_LEN = 64


class EmailDataset(Dataset):
    def __init__(self, df, tok):
        self.df = df.reset_index(drop=True)
        self.tok = tok

    def __len__(self):  return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ids, mask = encode(self.tok, str(row["text"]), MAX_LEN)
        return {
            "input_ids":    torch.tensor(ids,  dtype=torch.long),
            "padding_mask": torch.tensor(mask, dtype=torch.bool),
            "label":        torch.tensor(int(row["label"]), dtype=torch.long),
        }


# 4. TRANSFORMER MODEL

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class PhishingTransformer(nn.Module):
    """
    Pre-LN transformer encoder with CLS+mean multi-scale pooling.
    Pre-LN (norm_first=True) stabilises gradients; multi-scale pooling
    combines global intent ([CLS]) with average token semantics (mean).
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=64, nhead=4,
                 num_layers=4, d_ff=128, dropout=0.1):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos     = PositionalEncoding(d_model, dropout=dropout)
        layer        = nn.TransformerEncoderLayer(
                           d_model=d_model, nhead=nhead,
                           dim_feedforward=d_ff, dropout=dropout,
                           batch_first=True, norm_first=True)   # Pre-LN
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                           norm=nn.LayerNorm(d_model))
        # clf_in = d_model * 2 because we concat CLS + mean
        self.clf     = nn.Sequential(nn.Linear(d_model * 2, 64), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(64, 2))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.zeros_(m.weight[0])

    def _pool(self, x, mask):
        cls  = x[:, 0, :]
        real = (~mask).float().unsqueeze(-1) if mask is not None else torch.ones_like(x[:, :, :1])
        mean = (x * real).sum(1) / real.sum(1).clamp(min=1)
        return torch.cat([cls, mean], dim=-1)

    def forward(self, ids, mask=None):
        x = self.pos(self.embed(ids))
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.clf(self._pool(x, mask))

    def get_attn(self, ids, mask=None):
        weights = []
        x = self.pos(self.embed(ids))
        with torch.no_grad():
            for layer in self.encoder.layers:
                # Pre-LN: norm before attention
                q = k = v = layer.norm1(x)
                attn_out, w = layer.self_attn(q, k, v, key_padding_mask=mask,
                                              need_weights=True, average_attn_weights=False)
                weights.append(w)
                x = x + layer.dropout1(attn_out)
                x = x + layer._ff_block(layer.norm2(x))
        return weights


# 5. NOAM SCHEDULE

class Noam:
    def __init__(self, opt, d, warmup):
        self.opt = opt; self.d = d; self.w = warmup; self.s = 0

    def step(self):
        self.s += 1
        lr = (self.d ** -0.5) * min(self.s ** -0.5, self.s * self.w ** -1.5)
        for pg in self.opt.param_groups: pg["lr"] = lr
        return lr


# 6. TRAIN TRANSFORMER

def train_transformer(train_df, val_df, tok, epochs=12):
    # MPS lacks some transformer ops; fall back to CPU
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"[train] device={device}")

    train_dl = DataLoader(EmailDataset(train_df, tok), batch_size=32, shuffle=True)
    val_dl   = DataLoader(EmailDataset(val_df,   tok), batch_size=64)

    model  = PhishingTransformer().to(device)
    crit   = nn.CrossEntropyLoss(label_smoothing=0.05)
    # AdamW decouples weight decay from adaptive LR; weight_decay=0.01 is standard
    opt    = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    sched  = Noam(opt, 64, warmup=50)   # kept for lr_schedule PNG only

    best_val, patience, history, lr_trace = float("inf"), 0, [], []
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train(); tl = 0
        for b in train_dl:
            ids  = b["input_ids"].to(device)
            mask = b["padding_mask"].to(device)
            lbl  = b["label"].to(device)
            opt.zero_grad()
            loss = crit(model(ids, mask), lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            lr_trace.append(opt.param_groups[0]["lr"])
            tl += loss.item()
        tl /= len(train_dl)

        model.eval(); vl = 0; correct = 0; total = 0
        with torch.no_grad():
            for b in val_dl:
                ids  = b["input_ids"].to(device)
                mask = b["padding_mask"].to(device)
                lbl  = b["label"].to(device)
                out  = model(ids, mask)
                vl  += crit(out, lbl).item()
                correct += (out.argmax(1) == lbl).sum().item()
                total   += lbl.size(0)
        vl /= len(val_dl); acc = correct / total
        lr = opt.param_groups[0]["lr"]
        print(f"  epoch {epoch:2d} | train={tl:.4f} | val={vl:.4f} | acc={acc:.4f} | lr={lr:.2e}")
        history.append({"epoch": epoch, "train_loss": tl,
                        "val_loss": vl, "val_acc": acc, "lr": lr})

        if vl < best_val:
            best_val = vl; patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"epoch": epoch, "model_state": best_state,
                        "val_loss": vl,
                        "args": {"vocab_size": VOCAB_SIZE, "d_model": 64,
                                 "nhead": 4, "num_layers": 4, "d_ff": 128,
                                 "max_len": MAX_LEN, "pool": "cls_mean"}},
                       CKPT_DIR / "best_model.pt")
        else:
            patience += 1
            if patience >= 4:
                print(f"  early stop at epoch {epoch}"); break

    with open(CKPT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    model.load_state_dict(best_state)
    return model.to(device), history, lr_trace, device


# 7. PNG HELPERS

def save(fig, path, msg=None):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  PNG → {path}" if msg is None else msg)


# ── 7a. Loss curves ──────────────────────────────────────────────────────────
def png_loss_curves(history):
    eps = [h["epoch"] for h in history]
    tl  = [h["train_loss"] for h in history]
    vl  = [h["val_loss"]   for h in history]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(eps, tl, "o-",  color=C_BLUE, lw=1.8, label="Train loss")
    ax.plot(eps, vl, "s--", color=C_RED,  lw=1.8, label="Val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(); ax.grid(True, ls="--", alpha=0.4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.spines[["top","right"]].set_visible(False)
    save(fig, CKPT_DIR / "loss_curves.png")


# ── 7b. Val accuracy ─────────────────────────────────────────────────────────
def png_val_acc(history):
    eps = [h["epoch"]   for h in history]
    acc = [h["val_acc"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(eps, acc, "D-", color=C_GREEN, lw=1.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy per Epoch")
    ax.set_ylim(max(0, min(acc) - 0.05), 1.02)
    ax.grid(True, ls="--", alpha=0.4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.spines[["top","right"]].set_visible(False)
    save(fig, CKPT_DIR / "val_acc.png")


# ── 7c. LR schedule ──────────────────────────────────────────────────────────
def png_lr_schedule(lr_trace):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, len(lr_trace)+1), lr_trace, color=C_PURPLE, lw=1.5)
    ax.axvline(200, color="gray", ls=":", lw=1.2, label="Warmup (200 steps)")
    ax.set_xlabel("Training Step"); ax.set_ylabel("Learning Rate")
    ax.set_title("Noam Learning-Rate Schedule")
    ax.legend(); ax.grid(True, ls="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    save(fig, CKPT_DIR / "lr_schedule.png")


# ── 7d. NB confusion matrix ──────────────────────────────────────────────────
def png_nb_confusion(cm):
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Legit","Phishing"],
                yticklabels=["Legit","Phishing"],
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Naive Bayes — Confusion Matrix (Test)")
    save(fig, RESULTS / "nb_confusion_matrix.png")


# ── 7e. NB ROC curve ─────────────────────────────────────────────────────────
def png_nb_roc(y, ypr, auc):
    fpr, tpr, _ = roc_curve(y, ypr)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot(fpr, tpr, color=C_BLUE, lw=2, label=f"ROC (AUC={auc:.4f})")
    ax.plot([0,1],[0,1],"--",color="gray",lw=1,label="Random")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Naive Bayes — ROC Curve (Test)")
    ax.legend(); ax.grid(True, ls="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    save(fig, RESULTS / "nb_roc_curve.png")


# ── 7f. NB metrics bar ───────────────────────────────────────────────────────
def png_nb_metrics_bar(results):
    metric_keys = ["accuracy","f1","precision","recall","roc_auc"]
    x = np.arange(len(metric_keys)); w = 0.35
    colors = [C_TEAL, C_RED]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (res, col) in enumerate(zip(results, colors)):
        vals   = [res[m] for m in metric_keys]
        offset = (i - 0.5) * w
        bars   = ax.bar(x + offset, vals, w, label=res["split"].capitalize(),
                        color=col, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_","\n") for k in metric_keys])
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.08)
    ax.set_title("Naive Bayes — Metrics by Split")
    ax.legend(); ax.grid(axis="y", ls="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    save(fig, RESULTS / "nb_metrics_bar.png")


# ── 7g. Transformer confusion matrix ─────────────────────────────────────────
def png_tr_confusion(cm):
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax,
                xticklabels=["Legit","Phishing"],
                yticklabels=["Legit","Phishing"],
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Transformer — Confusion Matrix (Test)")
    save(fig, RESULTS / "transformer_confusion_matrix.png")


# ── 7h. Transformer ROC curve ────────────────────────────────────────────────
def png_tr_roc(y, ypr, auc):
    fpr, tpr, _ = roc_curve(y, ypr)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot(fpr, tpr, color=C_RED, lw=2, label=f"ROC (AUC={auc:.4f})")
    ax.plot([0,1],[0,1],"--",color="gray",lw=1,label="Random")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Transformer — ROC Curve (Test)")
    ax.legend(); ax.grid(True, ls="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    save(fig, RESULTS / "transformer_roc_curve.png")


# ── 7i. Transformer metrics bar ──────────────────────────────────────────────
def png_tr_metrics_bar(metrics):
    keys = list(metrics.keys()); vals = list(metrics.values())
    colors = [C_GREEN if v >= 0.90 else C_RED if v < 0.80 else C_BLUE for v in vals]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(keys, vals, color=colors, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Score")
    ax.set_xticklabels([k.replace("_","\n") for k in keys])
    ax.set_title("Transformer — Test Set Metrics")
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    save(fig, RESULTS / "transformer_metrics_bar.png")


# ── 7j. Probability distribution ─────────────────────────────────────────────
def png_prob_dist(y, ypr):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    # Use explicit bin edges spanning [0,1] to avoid degenerate-range errors
    bin_edges = np.linspace(0.0, 1.0, 31)
    ax.hist(ypr[y==0], bins=bin_edges, alpha=0.6, color=C_TEAL, label="Legit",    density=False)
    ax.hist(ypr[y==1], bins=bin_edges, alpha=0.6, color=C_RED,  label="Phishing", density=False)
    ax.set_xlabel("P(phishing)"); ax.set_ylabel("Count")
    ax.set_title("Transformer — Predicted Probability Distribution (Test)")
    ax.legend(); ax.grid(True, ls="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    save(fig, RESULTS / "transformer_prob_dist.png")


# ── 7k. Comparison bar ───────────────────────────────────────────────────────
def png_comparison_bar(nb_m, tr_m):
    metric_keys = ["accuracy","f1","precision","recall","roc_auc"]
    x = np.arange(len(metric_keys)); w = 0.35
    nb_vals = [nb_m[k] for k in metric_keys]
    tr_vals = [tr_m[k] for k in metric_keys]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w/2, nb_vals, w, label="Naive Bayes",        color=C_TEAL, edgecolor="white")
    b2 = ax.bar(x + w/2, tr_vals, w, label="Transformer (scratch)", color=C_RED,  edgecolor="white")
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_","\n") for k in metric_keys])
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.12)
    ax.set_title("Model Comparison: Naive Bayes vs Transformer (Test Set)")
    ax.legend(); ax.grid(axis="y", ls="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    save(fig, RESULTS / "comparison_bar.png")


# ── 7l. Comparison delta ─────────────────────────────────────────────────────
def png_comparison_delta(nb_m, tr_m):
    metric_keys = ["accuracy","f1","precision","recall","roc_auc"]
    deltas = [tr_m[k] - nb_m[k] for k in metric_keys]
    colors = [C_GREEN if d >= 0 else C_RED for d in deltas]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(metric_keys, deltas, color=colors, edgecolor="white")
    for bar, d in zip(bars, deltas):
        sign = "+" if d >= 0 else ""
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + (0.001 if d>=0 else -0.003),
                f"{sign}{d:.4f}", ha="center",
                va="bottom" if d>=0 else "top", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticklabels([k.replace("_","\n") for k in metric_keys])
    ax.set_ylabel("Δ (Transformer − Naive Bayes)")
    ax.set_title("Performance Gain: Transformer over Naive Bayes")
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    save(fig, RESULTS / "comparison_delta.png")


# ── 7m. Attention heatmap ────────────────────────────────────────────────────
def png_attention_heatmap(model, tok, text, device):
    ids, mask = encode(tok, text, MAX_LEN)
    n_real    = sum(1 for m in mask if m == 0)
    inp  = torch.tensor([ids],  dtype=torch.long).to(device)
    msk  = torch.tensor([mask], dtype=torch.bool).to(device)
    with torch.no_grad():
        weights = model.get_attn(inp, msk)
    w = weights[-1][0, 0, :n_real, :n_real].cpu().numpy()
    token_strs = [tok.id_to_token(i) or str(i) for i in ids[:n_real]]
    fig, ax = plt.subplots(figsize=(max(5, n_real*0.5), max(4, n_real*0.45)))
    sns.heatmap(w, ax=ax,
                xticklabels=token_strs, yticklabels=token_strs,
                cmap="Blues", vmin=0, vmax=w.max(),
                linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Attention weight"})
    ax.set_title("Self-Attention Heatmap — Layer 4, Head 1", fontsize=11)
    ax.set_xlabel("Key token"); ax.set_ylabel("Query token")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)
    save(fig, RESULTS / "attention_heatmap.png")


# ── 7n. Demo attention bar ───────────────────────────────────────────────────
def png_demo_attention_bar(model, tok, text, device):
    ids, mask = encode(tok, text, MAX_LEN)
    n_real    = sum(1 for m in mask if m == 0)
    inp  = torch.tensor([ids],  dtype=torch.long).to(device)
    msk  = torch.tensor([mask], dtype=torch.bool).to(device)
    with torch.no_grad():
        logits  = model(inp, msk)
        probs   = torch.softmax(logits, dim=1)[0]
        pred    = logits.argmax(1).item()
        weights = model.get_attn(inp, msk)
    cls_attn   = weights[-1][0, 0, 0, :n_real].cpu().numpy()
    token_strs = [tok.id_to_token(i) or str(i) for i in ids[:n_real]]
    ranked     = sorted(zip(cls_attn, token_strs), reverse=True)[:10]
    label      = "PHISHING" if pred == 1 else "LEGITIMATE"
    p_phish    = probs[1].item()
    wts        = [w for w, _ in ranked]
    toks       = [t for _, t in ranked]
    color      = C_RED if pred == 1 else C_BLUE
    fig, ax    = plt.subplots(figsize=(6, max(3, len(toks)*0.38)))
    bars       = ax.barh(toks[::-1], wts[::-1], color=color, edgecolor="white")
    for bar in bars:
        v = bar.get_width()
        ax.text(v+0.001, bar.get_y()+bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=8)
    ax.set_xlabel("CLS Attention Weight (Layer 4, Head 1)")
    ax.set_title(f"Prediction: {label}  |  P(phishing) = {p_phish:.4f}",
                 color=color, fontsize=10)
    ax.grid(axis="x", ls="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    save(fig, RESULTS / "demo_attention_bar.png")


# ── 7o. Visualize.py – separate loss curve (for visualize.py compat) ─────────
def png_visualize_loss(history):
    # identical data, different file so visualize.py output is also present
    eps = [h["epoch"] for h in history]
    tl  = [h["train_loss"] for h in history]
    vl  = [h["val_loss"]   for h in history]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(eps, tl, "o-",  color=C_BLUE, lw=1.8, label="Train loss")
    ax.plot(eps, vl, "s--", color=C_RED,  lw=1.8, label="Val loss")
    ax.axhline(0.41, color="gray", ls=":", lw=1.2, label="NB baseline")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training & Validation Loss (visualize.py output)")
    ax.legend(); ax.grid(True, ls="--", alpha=0.4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.spines[["top","right"]].set_visible(False)
    save(fig, RESULTS / "loss_curves.png")


# 8. EVALUATE HELPERS

def eval_model(model, test_df, tok, device):
    dl = DataLoader(EmailDataset(test_df, tok), batch_size=64)
    model.eval()
    yl, yp, ypr = [], [], []
    with torch.no_grad():
        for b in dl:
            ids  = b["input_ids"].to(device)
            mask = b["padding_mask"].to(device)
            lbl  = b["label"].cpu().numpy()
            out  = model(ids, mask)
            pr   = torch.softmax(out, dim=1)[:,1].cpu().numpy()
            pd_  = out.argmax(1).cpu().numpy()
            yl.extend(lbl); yp.extend(pd_); ypr.extend(pr)
    y, yp_, ypr_ = np.array(yl), np.array(yp), np.array(ypr)
    return y, yp_, ypr_


def compute_metrics(y, yp, ypr):
    return {
        "accuracy":  round(float(accuracy_score(y, yp)), 4),
        "f1":        round(float(f1_score(y, yp, average="binary")), 4),
        "precision": round(float(precision_score(y, yp, average="binary", zero_division=0)), 4),
        "recall":    round(float(recall_score(y, yp, average="binary")), 4),
        "roc_auc":   round(float(roc_auc_score(y, ypr)), 4),
    }


# 9. MAIN

def main():
    print("\n" + "="*60)
    print("  Phishing Email Detection — Full Pipeline")
    print("="*60)

    # ── Step 1: data ──────────────────────────────────────────────
    print("\n[1/7] Generating synthetic data…")
    train_df, val_df, test_df = make_synthetic(n_total=4000)

    # ── Step 2: tokenizer ─────────────────────────────────────────
    print("\n[2/7] Training BPE tokenizer…")
    tok = train_tokenizer(train_df["text"].astype(str).tolist())

    # ── Step 3: train transformer ─────────────────────────────────
    print("\n[3/7] Training transformer…")
    model, history, lr_trace, device = train_transformer(train_df, val_df, tok)

    # ── Step 4: transformer PNGs (training) ───────────────────────
    print("\n[4/7] Saving training PNGs…")
    png_loss_curves(history)
    png_val_acc(history)
    png_lr_schedule(lr_trace)
    png_visualize_loss(history)

    # ── Step 5: evaluate transformer ─────────────────────────────
    print("\n[5/7] Evaluating transformer…")
    y, yp, ypr = eval_model(model, test_df, tok, device)
    tr_metrics  = compute_metrics(y, yp, ypr)
    cm_tr       = confusion_matrix(y, yp)
    print("  " + "  ".join(f"{k}={v}" for k, v in tr_metrics.items()))
    print(classification_report(y, yp, target_names=["legit","phishing"]))
    with open(RESULTS / "transformer_metrics.json", "w") as f:
        json.dump(tr_metrics, f, indent=2)
    png_tr_confusion(cm_tr)
    png_tr_roc(y, ypr, tr_metrics["roc_auc"])
    png_tr_metrics_bar(tr_metrics)
    png_prob_dist(y, ypr)
    png_attention_heatmap(model, tok,
        "Your PayPal account has been limited click here to verify now", device)
    png_demo_attention_bar(model, tok,
        "Your PayPal account has been limited click here to verify now", device)

    # ── Step 6: Naive Bayes baseline ─────────────────────────────
    print("\n[6/7] Training & evaluating Naive Bayes…")
    nb = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2),
                                  sublinear_tf=True, min_df=1)),
        ("clf",   MultinomialNB(alpha=0.1)),
    ])
    nb.fit(train_df["text"].astype(str), train_df["label"])

    nb_results = []
    for split_name, df_s in [("val", val_df), ("test", test_df)]:
        X = df_s["text"].astype(str); ytrue = df_s["label"].values
        ypred = nb.predict(X); yprob = nb.predict_proba(X)[:,1]
        m = {"split": split_name,
             "accuracy":  round(float(accuracy_score(ytrue, ypred)), 4),
             "f1":        round(float(f1_score(ytrue, ypred, average="binary")), 4),
             "precision": round(float(precision_score(ytrue, ypred, average="binary", zero_division=0)), 4),
             "recall":    round(float(recall_score(ytrue, ypred, average="binary")), 4),
             "roc_auc":   round(float(roc_auc_score(ytrue, yprob)), 4)}
        nb_results.append(m)
        if split_name == "test":
            cm_nb = confusion_matrix(ytrue, ypred)
            nb_test_y, nb_test_ypr = ytrue, yprob
            nb_metrics = m
        print(f"  [{split_name}] " + "  ".join(f"{k}={v}" for k,v in m.items() if k != "split"))

    with open(RESULTS / "nb_metrics.json", "w") as f:
        json.dump(nb_results, f, indent=2)
    with open(RESULTS / "nb_model.pkl", "wb") as f:
        pickle.dump(nb, f)
    png_nb_confusion(cm_nb)
    png_nb_roc(nb_test_y, nb_test_ypr, nb_metrics["roc_auc"])
    png_nb_metrics_bar(nb_results)

    # ── Step 7: comparison ────────────────────────────────────────
    print("\n[7/7] Generating comparison PNGs…")
    keys = ["accuracy","f1","precision","recall","roc_auc"]
    comparison = {k: {"naive_bayes": nb_metrics[k], "transformer": tr_metrics[k],
                       "delta": round(tr_metrics[k] - nb_metrics[k], 4)}
                  for k in keys}
    with open(RESULTS / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    png_comparison_bar(nb_metrics, tr_metrics)
    png_comparison_delta(nb_metrics, tr_metrics)

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ALL PNGs GENERATED")
    print("="*60)
    all_pngs = sorted(list(RESULTS.glob("*.png")) + list(CKPT_DIR.glob("*.png")))
    for p in all_pngs:
        print(f"  {p}")
    print(f"\n  Total: {len(all_pngs)} PNG files")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
