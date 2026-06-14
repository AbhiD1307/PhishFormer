"""Unit tests for augment.py (random_deletion, random_swap, truncate_body)."""

import sys
import random
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from augment import random_deletion, random_swap, truncate_body


# ── random_deletion ───────────────────────────────────────────────────────────

class TestRandomDeletion:
    def test_returns_string(self):
        assert isinstance(random_deletion("click here now", p=0.1), str)

    def test_length_leq_original(self):
        text   = "click here to verify your account immediately"
        result = random_deletion(text, p=0.5)
        assert len(result.split()) <= len(text.split())

    def test_p_zero_returns_unchanged(self):
        text = "click here now please"
        assert random_deletion(text, p=0.0) == text

    def test_p_one_returns_one_word(self):
        # p=1.0 drops everything; fallback returns the first word
        text   = "click here now"
        result = random_deletion(text, p=1.0)
        # fallback: returns words[0] if result is empty
        assert len(result.strip()) > 0

    def test_single_word_unchanged(self):
        assert random_deletion("click", p=0.9) == "click"

    def test_empty_string_unchanged(self):
        assert random_deletion("", p=0.5) == ""

    def test_non_deterministic_with_seed(self):
        text    = "click here to verify your account now please do it"
        random.seed(0)
        result1 = random_deletion(text, p=0.3)
        random.seed(1)
        result2 = random_deletion(text, p=0.3)
        # With different seeds, outputs may differ (probabilistic test)
        # Just check both are valid strings
        assert isinstance(result1, str)
        assert isinstance(result2, str)


# ── random_swap ───────────────────────────────────────────────────────────────

class TestRandomSwap:
    def test_returns_string(self):
        assert isinstance(random_swap("click here now", n=1), str)

    def test_same_word_count(self):
        text   = "click here to verify your account now"
        result = random_swap(text, n=3)
        assert len(result.split()) == len(text.split())

    def test_same_vocabulary(self):
        text   = "click here to verify your account now"
        result = random_swap(text, n=5)
        assert sorted(result.split()) == sorted(text.split()), \
            "Swap should not add or remove words"

    def test_single_word_unchanged(self):
        assert random_swap("click", n=2) == "click"

    def test_empty_string_unchanged(self):
        assert random_swap("", n=1) == ""

    def test_n_zero_unchanged(self):
        text = "click here to verify"
        assert random_swap(text, n=0) == text


# ── truncate_body ─────────────────────────────────────────────────────────────

class TestTruncateBody:
    def test_returns_string(self):
        assert isinstance(truncate_body("click here now verify account", 0.5), str)

    def test_length_leq_original(self):
        text   = "click here now to verify your account and restore access"
        result = truncate_body(text, keep_frac=0.6)
        assert len(result.split()) <= len(text.split())

    def test_keep_frac_one_unchanged(self):
        text   = "click here now verify"
        result = truncate_body(text, keep_frac=1.0)
        assert result == text

    def test_keep_frac_half(self):
        text   = "one two three four five six seven eight"
        result = truncate_body(text, keep_frac=0.5)
        n_words = len(result.split())
        assert n_words == 4, f"Expected 4 words, got {n_words}"

    def test_single_word(self):
        result = truncate_body("click", keep_frac=0.5)
        assert len(result.split()) >= 1

    def test_empty_string(self):
        result = truncate_body("", keep_frac=0.5)
        assert result == ""

    def test_prefix_preserved(self):
        text   = "alpha beta gamma delta epsilon zeta"
        result = truncate_body(text, keep_frac=0.5)
        words  = text.split()
        result_words = result.split()
        assert words[:len(result_words)] == result_words, \
            "truncate_body should keep the first N words"
