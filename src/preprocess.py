import os
import re
import email
import logging
import hashlib
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# Text cleaning

_HTML_TAG  = re.compile(r"<[^>]+>")
_URL       = re.compile(r"https?://\S+|www\.\S+")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = _HTML_TAG.sub(" ", text)
    # Replace URLs with a literal [URL] token instead of deleting them.
    # URL presence is one of the strongest phishing signals; stripping them
    # entirely throws away that information before the model ever sees it.
    text = _URL.sub(" [URL] ", text)
    text = text.lower()
    text = _WHITESPACE.sub(" ", text).strip()
    return text

# Enron loader  (maildir structure)

def load_enron(root: str, max_emails: int = 120_000) -> list[dict]:

    root = Path(root)
    records = []
    seen = set()

    for path in root.rglob("*."):
        if len(records) >= max_emails:
            break
        _parse_enron_file(path, records, seen)

    # Also handle flat files (no extension) common in Enron maildir
    for path in root.rglob("*"):
        if len(records) >= max_emails:
            break
        if path.is_file() and path.suffix == "":
            _parse_enron_file(path, records, seen)

    log.info("Enron: loaded %d emails", len(records))
    return records


def _parse_enron_file(path: Path, records: list, seen: set):
    try:
        raw = path.read_bytes()
        msg = email.message_from_bytes(raw)
        subject = msg.get("Subject", "") or ""
        body_parts = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body_parts.append(payload.decode("utf-8", errors="ignore"))
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body_parts.append(payload.decode("utf-8", errors="ignore"))

        combined = clean_text(subject + " " + " ".join(body_parts))
        if len(combined) < 20:
            return

        h = hashlib.md5(combined.encode()).hexdigest()
        if h in seen:
            return
        seen.add(h)
        records.append({"text": combined, "label": 0})
    except Exception:
        pass


# CEAS 2008 loader  (CSV with columns: label, date, sender, receiver, subject, body)

def load_ceas(path: str) -> list[dict]:

    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    df.columns = [c.lower().strip() for c in df.columns]

    records = []
    seen = set()

    if "subject" in df.columns and "body" in df.columns:
        for _, row in df.iterrows():
            subj = str(row.get("subject", "")) if pd.notna(row.get("subject", "")) else ""
            body = str(row.get("body", "")) if pd.notna(row.get("body", "")) else ""
            combined = clean_text(subj + " " + body)
            if len(combined) < 20:
                continue
            h = hashlib.md5(combined.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            lbl = int(row.get("label", 1))
            records.append({"text": combined, "label": lbl})
    elif "text" in df.columns and "label" in df.columns:
        for _, row in df.iterrows():
            combined = clean_text(str(row["text"]))
            if len(combined) < 20:
                continue
            h = hashlib.md5(combined.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            records.append({"text": combined, "label": int(row["label"])})
    else:
        raise ValueError(f"Unrecognised CEAS CSV columns: {list(df.columns)}")

    log.info("CEAS: loaded %d emails", len(records))
    return records


# Balance + split

def balance_and_split(
    legit: list[dict],
    phish: list[dict],
    ratio_legit: float = 0.60,
    total: int = 82_500,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_legit = int(total * ratio_legit)
    n_phish = total - n_legit

    legit_sample = resample(legit, n_samples=min(n_legit, len(legit)),
                            replace=False, random_state=seed)
    phish_sample = resample(phish, n_samples=min(n_phish, len(phish)),
                            replace=len(phish) < n_phish, random_state=seed)

    df = pd.DataFrame(legit_sample + phish_sample).sample(
        frac=1, random_state=seed).reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df, test_size=val_frac + test_frac, random_state=seed, stratify=df["label"])
    val_df, test_df = train_test_split(
        temp_df, test_size=test_frac / (val_frac + test_frac),
        random_state=seed, stratify=temp_df["label"])

    log.info("Split — train: %d  val: %d  test: %d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df

# CLI entry point

def main():
    parser = argparse.ArgumentParser(description="Preprocess phishing dataset")
    parser.add_argument("--enron_dir", default="data/enron_mail_20150507/maildir",
                        help="Root of the Enron maildir tree")
    parser.add_argument("--ceas_csv", default="data/ceas_2008.csv",
                        help="Path to CEAS 2008 CSV file")
    parser.add_argument("--out_dir", default="data/processed",
                        help="Output directory for train/val/test CSVs")
    parser.add_argument("--total", type=int, default=82_500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    legit = load_enron(args.enron_dir)
    phish = load_ceas(args.ceas_csv)

    train_df, val_df, test_df = balance_and_split(
        legit, phish, total=args.total, seed=args.seed)

    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)
    log.info("Saved CSVs to %s", out)


if __name__ == "__main__":
    main()
