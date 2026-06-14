"""Unit tests for tokenizer.py (BPE encode/decode, special tokens, padding)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from tokenizer import train_tokenizer, load_tokenizer, encode, SPECIAL_TOKENS


# ── Fixtures ──────────────────────────────────────────────────────────────────

CORPUS = [
    "your paypal account has been suspended click here to verify",
    "hi alice please find the meeting notes attached from monday",
    "urgent your bank account will be closed verify details now",
    "reminder team standup tomorrow at 10am in conference room b",
    "congratulations you have won a prize claim your reward today",
    "the quarterly report for q2 2024 is now available for review",
    "security alert unusual login detected on your google account",
    "could you review the attached contract before end of week",
]


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    save_path = str(tmp_path_factory.mktemp("tok") / "tokenizer.json")
    tok = train_tokenizer(CORPUS, vocab_size=200, save_path=save_path)
    return tok


@pytest.fixture(scope="module")
def saved_tokenizer(tmp_path_factory):
    save_path = str(tmp_path_factory.mktemp("tok2") / "tokenizer.json")
    train_tokenizer(CORPUS, vocab_size=200, save_path=save_path)
    return load_tokenizer(save_path)


# ── Special tokens ────────────────────────────────────────────────────────────

class TestSpecialTokens:
    def test_pad_is_id_zero(self, tokenizer):
        assert tokenizer.token_to_id("[PAD]") == 0

    def test_cls_exists(self, tokenizer):
        assert tokenizer.token_to_id("[CLS]") is not None

    def test_sep_exists(self, tokenizer):
        assert tokenizer.token_to_id("[SEP]") is not None

    def test_unk_exists(self, tokenizer):
        assert tokenizer.token_to_id("[UNK]") is not None

    def test_all_special_tokens_present(self, tokenizer):
        for tok in SPECIAL_TOKENS:
            assert tokenizer.token_to_id(tok) is not None, \
                f"Special token {tok} missing from vocabulary"


# ── Encoding ──────────────────────────────────────────────────────────────────

class TestEncode:
    def test_output_length_equals_max_len(self, tokenizer):
        for max_len in [16, 32, 64]:
            ids, mask = encode(tokenizer, "click here to verify now", max_len)
            assert len(ids)  == max_len, f"ids length should be {max_len}"
            assert len(mask) == max_len, f"mask length should be {max_len}"

    def test_mask_bool_values(self, tokenizer):
        ids, mask = encode(tokenizer, "test email text", max_len=32)
        assert all(m in (0, 1) for m in mask), "Mask should contain only 0s and 1s"

    def test_padding_positions_are_zero(self, tokenizer):
        ids, mask = encode(tokenizer, "hi", max_len=32)
        for i, (tok_id, is_pad) in enumerate(zip(ids, mask)):
            if is_pad:
                assert tok_id == 0, f"Padded position {i} should be 0, got {tok_id}"

    def test_real_tokens_are_nonzero(self, tokenizer):
        ids, mask = encode(tokenizer, "click here to verify now", max_len=32)
        for i, (tok_id, is_pad) in enumerate(zip(ids, mask)):
            if not is_pad:
                assert tok_id != 0 or i == 0, \
                    f"Real token at position {i} should not be PAD (0)"

    def test_cls_token_is_first(self, tokenizer):
        ids, mask = encode(tokenizer, "test text", max_len=32)
        cls_id = tokenizer.token_to_id("[CLS]")
        assert ids[0] == cls_id, f"First token should be [CLS] (id={cls_id}), got {ids[0]}"

    def test_truncation(self, tokenizer):
        long_text = " ".join(["click"] * 200)
        ids, mask = encode(tokenizer, long_text, max_len=32)
        assert len(ids) == 32
        assert len(mask) == 32

    def test_empty_string(self, tokenizer):
        ids, mask = encode(tokenizer, "", max_len=16)
        assert len(ids) == 16
        assert len(mask) == 16

    def test_mask_prefix_false_then_true(self, tokenizer):
        """Real tokens come before padding — mask should be 0s then 1s."""
        ids, mask = encode(tokenizer, "verify now", max_len=32)
        seen_pad = False
        for m in mask:
            if m == 1:
                seen_pad = True
            if seen_pad:
                assert m == 1, "Once padding starts, all remaining positions must be 1"


# ── Save / load round-trip ────────────────────────────────────────────────────

class TestSaveLoad:
    def test_loaded_tokenizer_same_output(self, tokenizer, saved_tokenizer):
        text = "your account has been suspended"
        ids1, mask1 = encode(tokenizer,       text, max_len=32)
        ids2, mask2 = encode(saved_tokenizer, text, max_len=32)
        assert ids1  == ids2,  "Loaded tokenizer produces different ids"
        assert mask1 == mask2, "Loaded tokenizer produces different mask"

    def test_loaded_vocab_size_matches(self, tokenizer, saved_tokenizer):
        assert tokenizer.get_vocab_size() == saved_tokenizer.get_vocab_size()
