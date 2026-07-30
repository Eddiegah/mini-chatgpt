"""
Unit tests for the BPE tokenizer.

Tests:
  1. encode → decode round-trip on various text samples.
  2. Vocabulary size is exactly 256 + num_merges after training.
  3. All token IDs in an encoded sequence are valid vocabulary entries.
  4. Round-trip on tricky cases: punctuation, numbers, whitespace, unicode.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import BPETokenizer


# ── Fixture: tokenizer trained on a small sample corpus ─────────────────

@pytest.fixture(scope="module")
def trained_tokenizer():
    """
    Build a tokenizer trained on a small but representative corpus.
    Using scope='module' so training only runs once for all tests.
    """
    # Sample corpus: a snippet of Shakespeare to test realistic usage
    corpus = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them. To die: to sleep;
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to, 'tis a consummation
    Devoutly to be wish'd. To die, to sleep;
    To sleep! perchance to dream:
    HAMLET: All the world's a stage,
    And all the men and women merely players.
    They have their exits and their entrances,
    And one man in his time plays many parts,
    His acts being seven ages.
    1234567890 !@#$%^&*() Unicode: café, résumé, naïve, 日本語
    """ * 10  # Repeat to give BPE enough data to learn merges

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, num_merges=100, verbose=False)
    return tokenizer


# ── Tests ────────────────────────────────────────────────────────────────

class TestRoundTrip:
    """encode → decode must reproduce the original text exactly."""

    def test_simple_ascii(self, trained_tokenizer):
        text = "Hello, world!"
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text

    def test_longer_text(self, trained_tokenizer):
        text = "To be or not to be, that is the question."
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text

    def test_punctuation_heavy(self, trained_tokenizer):
        text = "What?! Really... I can't believe it; this (and that) is 'surprising'."
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text

    def test_numbers_and_symbols(self, trained_tokenizer):
        text = "Price: $3.99 (was $5.00) — 20% off! Order #1234."
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text

    def test_whitespace_preservation(self, trained_tokenizer):
        text = "Word1   Word2\tTabbed\n  Indented\n\nDouble newline"
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text

    def test_unicode_accented(self, trained_tokenizer):
        text = "café résumé naïve façade"
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text

    def test_single_character(self, trained_tokenizer):
        for char in ["a", " ", "\n", ".", "!"]:
            assert trained_tokenizer.decode(trained_tokenizer.encode(char)) == char

    def test_empty_string(self, trained_tokenizer):
        assert trained_tokenizer.decode(trained_tokenizer.encode("")) == ""

    def test_multiline_paragraph(self, trained_tokenizer):
        text = (
            "HAMLET: To be, or not to be, that is the question:\n"
            "Whether 'tis nobler in the mind to suffer\n"
            "The slings and arrows of outrageous fortune."
        )
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text

    def test_repeated_characters(self, trained_tokenizer):
        text = "aaaaabbbbbccccc"
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text


class TestVocabulary:
    """Verify vocabulary size and token ID validity."""

    def test_vocab_size_correct(self, trained_tokenizer):
        # vocab_size should be exactly 256 base bytes + 100 merges = 356
        assert trained_tokenizer.vocab_size == 256 + 100, (
            f"Expected vocab_size=356, got {trained_tokenizer.vocab_size}"
        )

    def test_all_base_bytes_in_vocab(self, trained_tokenizer):
        # All 256 byte values must be in the vocabulary
        for i in range(256):
            assert i in trained_tokenizer.vocab, f"Byte {i} missing from vocab"
            assert trained_tokenizer.vocab[i] == bytes([i]), (
                f"Vocab entry {i} should be bytes([{i}]), got {trained_tokenizer.vocab[i]!r}"
            )

    def test_encoded_ids_all_in_vocab(self, trained_tokenizer):
        text = "All the world's a stage, and all the men and women merely players."
        ids = trained_tokenizer.encode(text)
        for token_id in ids:
            assert token_id in trained_tokenizer.vocab, (
                f"Token ID {token_id} not in vocabulary"
            )

    def test_no_empty_tokens_in_encoding(self, trained_tokenizer):
        text = "Hello world"
        ids = trained_tokenizer.encode(text)
        assert len(ids) > 0, "Encoding non-empty text should not return empty list"


class TestSaveLoad:
    """Test that a tokenizer round-trips correctly through save/load."""

    def test_save_load_round_trip(self, trained_tokenizer, tmp_path):
        save_path = str(tmp_path / "test_tokenizer.json")
        trained_tokenizer.save(save_path)

        loaded = BPETokenizer()
        loaded.load(save_path)

        # Vocab should match
        assert loaded.vocab_size == trained_tokenizer.vocab_size
        assert loaded.merges == trained_tokenizer.merges

        # Encoding should be identical
        text = "All the world's a stage"
        assert loaded.encode(text) == trained_tokenizer.encode(text)

        # Round-trip should still work
        assert loaded.decode(loaded.encode(text)) == text
