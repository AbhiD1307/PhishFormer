"""
Batch inference for the PhishingTransformer.

Accepts a CSV file (with a 'text' column), a directory of .txt/.eml files,
or a single email string via --text, and writes a predictions CSV.

Usage examples:
    # Classify a CSV of emails
    python src/inference.py --input emails.csv --out results/predictions.csv

    # Classify a folder of raw .eml / .txt files
    python src/inference.py --input_dir data/raw_emails/ --out results/predictions.csv

    # Single email string (quick test)
    python src/inference.py --text "Your PayPal account is suspended. Click here now."

Output CSV columns:
    text, p_phishing, p_legit, prediction, confidence
"""

import argparse
import email
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import PhishingTransformer
from tokenizer import load_tokenizer, encode
from preprocess import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LABEL_MAP = {0: "LEGITIMATE", 1: "PHISHING"}


def _load_model(checkpoint: str, device: torch.device) -> tuple[PhishingTransformer, dict]:
    ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})
    model = PhishingTransformer(
        vocab_size  = saved.get("vocab_size", 16_000),
        d_model     = saved.get("d_model",    256),
        nhead       = saved.get("nhead",      8),
        num_layers  = saved.get("num_layers", 6),
        d_ff        = saved.get("d_ff",       1024),
        dropout     = 0.0,
        pool        = saved.get("pool",       "cls_mean"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info("Loaded checkpoint (epoch %d, val_loss=%.4f)",
             ckpt.get("epoch", -1), ckpt.get("val_loss", float("nan")))
    return model, saved


def _parse_eml(path: Path) -> str:
    """Extract plain-text body from a .eml file."""
    try:
        raw = path.read_bytes()
        msg = email.message_from_bytes(raw)
        subject = msg.get("Subject", "") or ""
        parts   = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        parts.append(payload.decode("utf-8", errors="ignore"))
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                parts.append(payload.decode("utf-8", errors="ignore"))
        return clean_text(subject + " " + " ".join(parts))
    except Exception as e:
        log.warning("Could not parse %s: %s", path, e)
        return ""


def _texts_from_dir(input_dir: str) -> list[str]:
    """Read all .txt and .eml files from a directory."""
    texts = []
    for path in sorted(Path(input_dir).rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".eml":
            texts.append(_parse_eml(path))
        elif path.suffix in (".txt", ""):
            texts.append(clean_text(path.read_text(errors="ignore")))
    log.info("Loaded %d files from %s", len(texts), input_dir)
    return texts


@torch.no_grad()
def _run_inference(
    texts:        list[str],
    model:        PhishingTransformer,
    tok,
    max_len:      int,
    batch_size:   int,
    device:       torch.device,
    threshold:    float,
) -> pd.DataFrame:
    all_p_phish, all_p_legit, all_pred, all_conf = [], [], [], []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]

        ids_list, mask_list = [], []
        for t in batch_texts:
            ids, mask = encode(tok, t, max_len)
            ids_list.append(ids)
            mask_list.append(mask)

        input_ids    = torch.tensor(ids_list,  dtype=torch.long).to(device)
        padding_mask = torch.tensor(mask_list, dtype=torch.bool).to(device)

        logits = model(input_ids, padding_mask)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()

        p_phish = probs[:, 1]
        p_legit = probs[:, 0]
        pred    = (p_phish >= threshold).astype(int)
        conf    = np.where(pred == 1, p_phish, p_legit)

        all_p_phish.extend(p_phish.tolist())
        all_p_legit.extend(p_legit.tolist())
        all_pred.extend([LABEL_MAP[p] for p in pred.tolist()])
        all_conf.extend(conf.tolist())

    return pd.DataFrame({
        "text":       texts,
        "p_phishing": [round(v, 4) for v in all_p_phish],
        "p_legit":    [round(v, 4) for v in all_p_legit],
        "prediction": all_pred,
        "confidence": [round(v, 4) for v in all_conf],
    })


def infer(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    log.info("Device: %s", device)

    model, saved = _load_model(args.checkpoint, device)
    tok    = load_tokenizer(args.tokenizer)
    max_len = saved.get("max_len", args.max_len)

    # ── Gather texts ──────────────────────────────────────────────────────────
    if args.text:
        texts = [clean_text(args.text)]
    elif args.input_dir:
        texts = _texts_from_dir(args.input_dir)
    elif args.input:
        df    = pd.read_csv(args.input)
        if "text" not in df.columns:
            raise ValueError(f"CSV must have a 'text' column. Found: {list(df.columns)}")
        texts = df["text"].astype(str).tolist()
        log.info("Loaded %d emails from %s", len(texts), args.input)
    else:
        raise ValueError("Provide --text, --input <csv>, or --input_dir <dir>")

    texts = [t for t in texts if t.strip()]
    if not texts:
        log.warning("No valid text found — nothing to classify.")
        return

    log.info("Classifying %d emails (threshold=%.2f)…", len(texts), args.threshold)

    # ── Run inference ─────────────────────────────────────────────────────────
    results = _run_inference(texts, model, tok, max_len, args.batch_size, device, args.threshold)

    # ── Print summary ─────────────────────────────────────────────────────────
    n_phish = (results["prediction"] == "PHISHING").sum()
    n_legit = len(results) - n_phish
    print(f"\n{'='*52}")
    print(f"  Classified : {len(results)} emails")
    print(f"  PHISHING   : {n_phish}  ({100*n_phish/len(results):.1f}%)")
    print(f"  LEGITIMATE : {n_legit}  ({100*n_legit/len(results):.1f}%)")
    print(f"{'='*52}")

    if args.text:
        row = results.iloc[0]
        print(f"\n  Prediction  : {row['prediction']}")
        print(f"  P(phishing) : {row['p_phishing']:.4f}")
        print(f"  P(legit)    : {row['p_legit']:.4f}")
        print(f"  Confidence  : {row['confidence']:.4f}")
    else:
        # Show top-5 most confident phishing detections
        phish_rows = results[results["prediction"] == "PHISHING"].nlargest(5, "confidence")
        if not phish_rows.empty:
            print("\n  Top-5 most confident PHISHING detections:")
            for _, r in phish_rows.iterrows():
                snippet = r["text"][:80].replace("\n", " ")
                print(f"    [{r['confidence']:.3f}] {snippet}…")

    # ── Save output ───────────────────────────────────────────────────────────
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.out, index=False)
        log.info("Predictions saved → %s", args.out)
    else:
        print("\n(Use --out <path.csv> to save results)")


def parse_args():
    p = argparse.ArgumentParser(description="Batch phishing email inference")
    # Input (mutually exclusive)
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--text",      type=str, help="Single email string to classify")
    grp.add_argument("--input",     type=str, help="CSV file with 'text' column")
    grp.add_argument("--input_dir", type=str, help="Directory of .txt/.eml files")
    # Model
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--tokenizer",  default="data/tokenizer.json")
    p.add_argument("--max_len",    type=int,   default=512)
    # Inference
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--threshold",  type=float, default=0.50,
                   help="P(phishing) threshold for PHISHING label (default 0.5)")
    # Output
    p.add_argument("--out", type=str, default=None,
                   help="Path to save predictions CSV (optional)")
    return p.parse_args()


if __name__ == "__main__":
    infer(parse_args())
