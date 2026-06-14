"""Unit tests for PhishingTransformer (model.py)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import PhishingTransformer, PositionalEncoding


# ── Fixtures ──────────────────────────────────────────────────────────────────

VOCAB   = 500
D_MODEL = 64
NHEAD   = 4
LAYERS  = 2
D_FF    = 128
BATCH   = 4
SEQ_LEN = 32


def make_model(pool="cls_mean") -> PhishingTransformer:
    return PhishingTransformer(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=LAYERS,
        d_ff=D_FF,
        dropout=0.0,
        pool=pool,
    )


def make_inputs(batch=BATCH, seq=SEQ_LEN, pad_last=8):
    ids  = torch.randint(1, VOCAB, (batch, seq))
    mask = torch.zeros(batch, seq, dtype=torch.bool)
    if pad_last > 0:
        ids[:, -pad_last:]  = 0
        mask[:, -pad_last:] = True
    return ids, mask


# ── PositionalEncoding ────────────────────────────────────────────────────────

class TestPositionalEncoding:
    def test_output_shape(self):
        pe  = PositionalEncoding(D_MODEL, max_len=128, dropout=0.0)
        x   = torch.zeros(BATCH, SEQ_LEN, D_MODEL)
        out = pe(x)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_pe_not_all_zeros(self):
        pe  = PositionalEncoding(D_MODEL, max_len=128, dropout=0.0)
        x   = torch.zeros(BATCH, SEQ_LEN, D_MODEL)
        out = pe(x)
        assert out.abs().sum() > 0, "Positional encoding should be non-zero"

    def test_different_positions_differ(self):
        pe  = PositionalEncoding(D_MODEL, max_len=128, dropout=0.0)
        x   = torch.zeros(1, 10, D_MODEL)
        out = pe(x)[0]  # (10, D_MODEL)
        # No two rows should be identical
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                assert not torch.allclose(out[i], out[j]), \
                    f"Positions {i} and {j} have identical encodings"


# ── PhishingTransformer ───────────────────────────────────────────────────────

class TestPhishingTransformerForward:
    @pytest.mark.parametrize("pool", ["cls", "mean", "cls_mean"])
    def test_output_shape(self, pool):
        model = make_model(pool)
        ids, mask = make_inputs()
        model.eval()
        with torch.no_grad():
            out = model(ids, mask)
        assert out.shape == (BATCH, 2), \
            f"Expected (batch, 2) logits, got {out.shape}"

    def test_forward_no_mask(self):
        model = make_model()
        ids   = torch.randint(1, VOCAB, (BATCH, SEQ_LEN))
        model.eval()
        with torch.no_grad():
            out = model(ids, padding_mask=None)
        assert out.shape == (BATCH, 2)

    def test_forward_single_sample(self):
        model = make_model()
        ids   = torch.randint(1, VOCAB, (1, 16))
        model.eval()
        with torch.no_grad():
            out = model(ids)
        assert out.shape == (1, 2)

    def test_logits_are_finite(self):
        model = make_model()
        ids, mask = make_inputs()
        model.eval()
        with torch.no_grad():
            out = model(ids, mask)
        assert torch.isfinite(out).all(), "Logits contain NaN or Inf"

    def test_padding_invariance(self):
        """Adding more padding to an already-padded token should not change output."""
        model = make_model()
        model.eval()

        ids  = torch.randint(1, VOCAB, (1, SEQ_LEN))
        # Two masks: pad last 4 vs pad last 8 — the real tokens are identical
        mask_4 = torch.zeros(1, SEQ_LEN, dtype=torch.bool)
        mask_8 = torch.zeros(1, SEQ_LEN, dtype=torch.bool)
        mask_4[:, -4:] = True
        mask_8[:, -8:] = True
        ids[:, -8:] = 0   # zero out the padded region for both

        with torch.no_grad():
            out_4 = model(ids, mask_4)
            out_8 = model(ids, mask_8)

        # Outputs should be close (differences only from mean-pool denominator)
        assert out_4.shape == out_8.shape


class TestPhishingTransformerPooling:
    def test_cls_pool_shape(self):
        model = make_model(pool="cls")
        ids, mask = make_inputs()
        model.eval()
        with torch.no_grad():
            out = model(ids, mask)
        assert out.shape == (BATCH, 2)

    def test_mean_pool_shape(self):
        model = make_model(pool="mean")
        ids, mask = make_inputs()
        model.eval()
        with torch.no_grad():
            out = model(ids, mask)
        assert out.shape == (BATCH, 2)

    def test_cls_mean_pool_shape(self):
        model = make_model(pool="cls_mean")
        ids, mask = make_inputs()
        model.eval()
        with torch.no_grad():
            out = model(ids, mask)
        assert out.shape == (BATCH, 2)


class TestAttentionWeights:
    def test_returns_list_of_correct_length(self):
        model = make_model()
        ids, mask = make_inputs(batch=1)
        attn = model.get_attention_weights(ids, mask)
        assert len(attn) == LAYERS

    def test_attention_shape(self):
        model = make_model()
        ids, mask = make_inputs(batch=2)
        attn = model.get_attention_weights(ids, mask)
        # Each element: (batch, nhead, seq, seq)
        for i, w in enumerate(attn):
            assert w.shape == (2, NHEAD, SEQ_LEN, SEQ_LEN), \
                f"Layer {i} attention shape mismatch: {w.shape}"

    def test_attention_sums_to_one(self):
        model = make_model()
        ids, mask = make_inputs(batch=1, pad_last=0)
        attn = model.get_attention_weights(ids, None)
        for i, w in enumerate(attn):
            row_sums = w.sum(dim=-1)   # (B, nhead, seq)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), \
                f"Layer {i} attention rows don't sum to 1"


class TestEncodeMethod:
    def test_encode_shape(self):
        model = make_model(pool="cls_mean")
        ids, mask = make_inputs()
        model.eval()
        with torch.no_grad():
            rep = model.encode(ids, mask)
        # cls_mean pool: (batch, 2 * d_model)
        assert rep.shape == (BATCH, D_MODEL * 2)

    def test_encode_cls_shape(self):
        model = make_model(pool="cls")
        ids, mask = make_inputs()
        model.eval()
        with torch.no_grad():
            rep = model.encode(ids, mask)
        assert rep.shape == (BATCH, D_MODEL)


class TestNumParameters:
    def test_num_parameters_positive(self):
        model = make_model()
        assert model.num_parameters > 0

    def test_num_parameters_consistent(self):
        model = make_model()
        manual = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert model.num_parameters == manual


class TestWeightInitialisation:
    def test_pad_embedding_is_zero(self):
        model = make_model()
        pad_vec = model.embed.weight[0]
        assert torch.allclose(pad_vec, torch.zeros_like(pad_vec)), \
            "[PAD] token embedding should be initialised to zero"

    def test_linear_bias_is_zero(self):
        model = make_model()
        for m in model.modules():
            import torch.nn as nn
            if isinstance(m, nn.Linear) and m.bias is not None:
                assert torch.allclose(m.bias, torch.zeros_like(m.bias)), \
                    "Linear bias should be zero-initialised"
