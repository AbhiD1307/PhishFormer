# PhishFormer — Phishing Email Detection via From-Scratch Transformer Encoder

**CSS 586 · Deep Learning & AI · University of Washington Bothell**
Abhishek Deshmukh · Spring 2026

---

## What this project does

Binary classifier that distinguishes phishing emails from legitimate ones.

- **PhishFormer** — 6-layer Pre-LN Transformer encoder (7.1M params) built from scratch in PyTorch, trained on 82,500 emails (Enron + CEAS 2008)
- **Naive Bayes baseline** — TF-IDF + Multinomial NB for comparison
- **Soft-voting ensemble** — combines both models; achieves F1 = 0.929
- **Explainability** — gradient saliency, attention rollout, per-head heatmaps
- **5-fold cross-validation** — confirms stability (F1 = 0.922 ± 0.003)
- **Hyperparameter search** — grid or random search over key architecture dims

### Key results (held-out test set, 8,250 emails)

| Model                  | Accuracy | F1    | Precision | Recall | ROC-AUC |
|------------------------|----------|-------|-----------|--------|---------|
| Naive Bayes (TF-IDF)   | 0.883    | 0.862 | 0.881     | 0.844  | 0.912   |
| **PhishFormer (ours)** | **0.931**| **0.924** | **0.911** | **0.937** | **0.963** |
| Ensemble (NB + Transformer) | 0.934 | **0.929** | 0.916 | 0.942 | **0.967** |

PhishFormer reduces missed phishing emails (false negatives) by **59%** vs Naive Bayes.

---

## Project structure

```
Project/
├── src/
│   ├── preprocess.py          # Load Enron + CEAS, clean, deduplicate, split
│   ├── tokenizer.py           # BPE tokenizer (16K vocab, [CLS]/[SEP]/[URL] tokens)
│   ├── dataset.py             # PyTorch Dataset + DataLoader factories
│   ├── augment.py             # On-the-fly email augmentation (delete/swap/truncate)
│   ├── model.py               # PhishingTransformer: Pre-LN, CLS+mean pooling
│   ├── train.py               # AdamW + Noam LR + AMP + label smoothing + augmentation
│   ├── baseline_nb.py         # Multinomial NB + TF-IDF baseline
│   ├── evaluate.py            # Test-set evaluation + confusion matrix + ROC
│   ├── visualize.py           # Loss curves + attention heatmap CLI
│   ├── compare_models.py      # Side-by-side NB vs Transformer metric table
│   ├── ensemble.py            # Soft-voting ensemble with weight sweep
│   ├── explain.py             # Gradient saliency + attention rollout + all-head heatmap
│   ├── cross_validate.py      # 5-fold stratified CV for both models
│   ├── inference.py           # Batch inference: CSV / folder of .eml / single string
│   ├── hyperparameter_search.py  # Grid or random search over architecture dims
│   ├── demo.py                # Single-email CLI prediction with attention display
│   └── tests/
│       ├── test_model.py      # Unit tests: forward pass, pooling, attention weights
│       ├── test_tokenizer.py  # Unit tests: encode/decode, special tokens, padding
│       └── test_augment.py    # Unit tests: augmentation functions
├── data/
│   ├── processed/             # Generated: train.csv, val.csv, test.csv
│   └── tokenizer.json         # Generated: trained BPE tokenizer
├── checkpoints/
│   ├── best_model.pt          # Generated: best checkpoint (saved by val loss)
│   ├── history.json           # Generated: per-epoch loss/acc log
│   ├── loss_curves.png
│   ├── val_acc.png
│   └── lr_schedule.png
├── results/
│   ├── nb_metrics.json
│   ├── transformer_metrics.json
│   ├── comparison.json
│   ├── ensemble/              # ensemble_results.json, ensemble_roc.png
│   ├── cv/                    # cv_results.json
│   ├── explain/               # saliency_bar.png, rollout_bar.png, all_heads_heatmap.png
│   └── hparam_search/         # hparam_results.json, hparam_search_bar.png
├── run_all.py                 # Self-contained end-to-end demo (synthetic data, no download)
├── requirements.txt
└── final_report.tex           # ACM-format conference paper
```

---

## Setup

```bash
# Python 3.11+ required
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quick demo (no data download needed)

`run_all.py` generates synthetic email data, trains everything, and produces all plots in one command:

```bash
python run_all.py
```

This takes ~2 minutes on CPU and verifies the full pipeline works.

---

## Full pipeline (real datasets)

### 1. Download datasets

| Dataset | URL |
|---------|-----|
| Enron Email Corpus | https://www.cs.cmu.edu/~enron/ → `enron_mail_20150507.tgz` |
| CEAS 2008 Spam Corpus | https://plg.uwaterloo.ca/~gvcormac/ceam08/ |

```
data/
├── enron_mail_20150507/maildir/   ← extract Enron here
└── ceas_2008.csv                  ← place CEAS CSV here
```

### 2. Preprocess

```bash
python src/preprocess.py \
    --enron_dir data/enron_mail_20150507/maildir \
    --ceas_csv  data/ceas_2008.csv \
    --out_dir   data/processed \
    --total     82500
```

### 3. Train BPE tokenizer

```bash
python src/tokenizer.py \
    --train_csv  data/processed/train.csv \
    --vocab_size 16000 \
    --out        data/tokenizer.json
```

### 4. Train Transformer

```bash
python src/train.py \
    --data_dir      data/processed \
    --tokenizer     data/tokenizer.json \
    --checkpoint_dir checkpoints \
    --d_model       256 \
    --nhead         8 \
    --num_layers    6 \
    --d_ff          1024 \
    --dropout       0.1 \
    --pool          cls_mean \
    --epochs        15 \
    --batch_size    32 \
    --warmup_steps  4000 \
    --patience      4 \
    --weight_decay  0.01 \
    --label_smoothing 0.05 \
    --augment
```

**Hardware:** ~3 h on Google Colab T4 GPU, ~25 min on Apple M2.

### 5. Train Naive Bayes baseline

```bash
python src/baseline_nb.py --data_dir data/processed --out_dir results
```

### 6. Evaluate Transformer

```bash
python src/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data_dir   data/processed \
    --tokenizer  data/tokenizer.json \
    --out_dir    results
```

### 7. Compare models

```bash
python src/compare_models.py \
    --nb_metrics          results/nb_metrics.json \
    --transformer_metrics results/transformer_metrics.json \
    --out_dir             results
```

---

## Ensemble

```bash
# Default weights (tr=0.65, nb=0.35)
python src/ensemble.py

# Auto-search best weights
python src/ensemble.py --sweep
```

Output saved to `results/ensemble/`.

---

## Cross-validation

```bash
python src/cross_validate.py --folds 5 --epochs 8
```

Output saved to `results/cv/cv_results.json`.

---

## Explainability

```bash
python src/explain.py \
    --text "Your PayPal account has been suspended. Verify now or lose access." \
    --method both \
    --out_dir results/explain
```

Produces:
- `saliency_bar.png` — gradient-based token importance (∂loss/∂embedding)
- `rollout_bar.png` — attention rollout from [CLS] through all layers
- `all_heads_heatmap.png` — full self-attention matrix for every head in the last layer

---

## Hyperparameter search

```bash
# Random search — 16 configs, 5 epochs each (~1 h on GPU)
python src/hyperparameter_search.py --mode random --n_trials 16 --epochs 5

# Full grid search (all combinations — slow)
python src/hyperparameter_search.py --mode grid --epochs 5
```

Output saved to `results/hparam_search/`.

---

## Batch inference (production-style)

```bash
# Classify a CSV of emails
python src/inference.py --input emails.csv --out results/predictions.csv

# Classify a folder of .eml / .txt files
python src/inference.py --input_dir data/raw_emails/ --out results/predictions.csv

# Quick single-email test
python src/inference.py \
    --text "Congratulations! You have won a prize. Click to claim."
```

---

## Attention visualization

```bash
# Loss curves
python src/visualize.py loss --history checkpoints/history.json

# Attention heatmap
python src/visualize.py attn \
    --checkpoint checkpoints/best_model.pt \
    --tokenizer  data/tokenizer.json \
    --text       "Your account has been suspended. Click here to verify." \
    --layer      5 --head 0
```

---

## Run tests

```bash
cd Project
python -m pytest src/tests/ -v
```

---

## Architecture

```
Input: Email token sequence (max 512 tokens, [CLS] prepended)
  ↓
Token Embedding (16K vocab, d=256) + Sinusoidal Positional Encoding
  ↓
Encoder Layer 1–6  (Pre-LN, 8-head MHA, d_ff=1024, GELU, Dropout=0.1)
  ↓
Multi-scale pooling: concat([CLS], mean(non-padding tokens)) → 512-dim
  ↓
Linear(512→256) → GELU → Dropout(0.1)
Linear(256→64)  → GELU
Linear(64→2)
  ↓
Output: P(phishing), P(legit)
```

**Total parameters: ~7.1M** (all randomly initialized — no pretrained weights)

**Key design choices:**
- **Pre-LN** (norm before each sub-layer): stable gradients, no LR sensitivity
- **Multi-scale pooling**: CLS captures global intent; mean captures lexical average
- **AdamW** (weight_decay=0.01): decouples weight decay from adaptive LR
- **Label smoothing** (ε=0.05): prevents overconfident predictions on noisy labels
- **[URL] token**: URLs replaced with literal token — preserves presence signal without memorising domains

---

## References

1. Vaswani et al., "Attention is All You Need," NeurIPS 2017
2. Devlin et al., "BERT," NAACL-HLT 2019
3. Loshchilov & Hutter, "Decoupled Weight Decay Regularization," ICLR 2019
4. Xiong et al., "On Layer Normalization in the Transformer Architecture," ICML 2020
5. Abnar & Zuidema, "Quantifying Attention Flow in Transformers," ACL 2020
6. Fette et al., "Learning to Detect Phishing Emails," WWW 2007
7. Klimt & Yang, "The Enron Corpus," ECML 2004
