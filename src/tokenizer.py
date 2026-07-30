"""
Byte-Pair Encoding (BPE) Tokenizer — from scratch.

BPE Algorithm Overview
-----------------------
BPE was originally a data-compression technique, adapted for NLP by Sennrich et al. (2016)
and used (in a slightly modified form) by GPT-2/3/4.

The core idea:
  1. Start with a vocabulary of individual bytes (256 tokens).  Working at the byte level
     means we can tokenize *any* UTF-8 text with zero unknown tokens — nothing ever falls
     outside the vocabulary.
  2. Scan the training corpus and find the most-frequent adjacent pair of tokens.
  3. Merge that pair into a single new token and add it to the vocabulary.
  4. Repeat steps 2–3 for `num_merges` iterations.

After training, we have a vocabulary of up to 256 + num_merges tokens and an ordered list
of merge rules.  Encoding new text means applying those same merges in the same order.

Why byte-level?
  - No "unknown token" problem.
  - Language-agnostic — works on Chinese, emoji, code, whatever.
  - Matches the approach used in the real GPT-2 tokenizer (though GPT-2 adds a regex
    pre-tokenization step to avoid merging across word boundaries, which we keep optional
    here for clarity).

Losslessness guarantee:
  encode() → list of ints
  decode() → original bytes → original text (exactly)
  This is verified by the unit test in tests/test_tokenizer.py.
"""

import collections
import json
import os
import regex  # 'regex' (not 're') supports \p{} Unicode categories

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# We represent every token as a Python int.
# Tokens 0-255 are the 256 raw byte values.
# Tokens 256+ are learned merge tokens.
PAD_ID   = 0   # We repurpose byte 0 as the pad token (null byte rarely appears in text)


# ---------------------------------------------------------------------------
# Helper: GPT-2-style pre-tokenization regex (optional but good practice)
# ---------------------------------------------------------------------------
# This pattern splits text at natural word/punctuation boundaries *before* BPE
# runs.  It prevents BPE from merging across word boundaries (e.g. "dog" at the
# end of a sentence never merges with the period that follows it).
# Taken directly from the GPT-2 paper / tiktoken source.
GPT2_PAT = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)|"""          # contractions: 's, 'd, 'm, 't, 'll, 've, 're
    r""" ?\p{L}+|"""                       # optional space + letters
    r""" ?\p{N}+|"""                       # optional space + digits
    r""" ?[^\s\p{L}\p{N}]+|"""            # optional space + punctuation/symbols
    r"""\s+(?!\S)|"""                      # trailing whitespace
    r"""\s+"""                             # other whitespace
)


def _text_to_byte_tokens(text: str) -> list[list[int]]:
    """
    Split `text` using the GPT-2 pattern, then convert each piece to a list of
    byte values (ints 0-255).  This is the starting point for BPE training.

    Example:
        "Hello!" → [["H","e","l","l","o","!"] as bytes]
                 = [[72, 101, 108, 108, 111, 33]]
    """
    words = regex.findall(GPT2_PAT, text)
    return [list(word.encode("utf-8")) for word in words]


def _get_pair_counts(vocab_tokens: list[list[int]]) -> dict[tuple[int, int], int]:
    """
    Count every adjacent pair across all token sequences.
    Returns {(a, b): frequency, ...}.
    """
    counts: dict[tuple[int, int], int] = collections.defaultdict(int)
    for seq in vocab_tokens:
        for a, b in zip(seq, seq[1:]):
            counts[(a, b)] += 1
    return counts


def _merge_pair(
    vocab_tokens: list[list[int]],
    pair: tuple[int, int],
    new_id: int,
) -> list[list[int]]:
    """
    Replace every occurrence of `pair` = (a, b) with `new_id` across all sequences.
    This is an in-place-style merge — we build new lists rather than mutating.

    Example:
        pair = (101, 108), new_id = 256
        [101, 108, 108, 111] → [256, 108, 111]
    """
    a, b = pair
    result = []
    for seq in vocab_tokens:
        new_seq: list[int] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                new_seq.append(new_id)
                i += 2  # skip both tokens of the pair
            else:
                new_seq.append(seq[i])
                i += 1
        result.append(new_seq)
    return result


# ---------------------------------------------------------------------------
# BPETokenizer class
# ---------------------------------------------------------------------------

class BPETokenizer:
    """
    A from-scratch Byte-Pair Encoding tokenizer.

    Attributes
    ----------
    vocab : dict[int, bytes]
        Maps token ID → raw bytes.  Byte tokens 0-255 map to their single byte.
        Merge tokens 256+ map to the concatenation of the bytes they represent.
    merges : list[tuple[int, int]]
        Ordered list of merge rules, in the order they were learned.
        During encoding, we apply these in order (earlier merges first).
    """

    def __init__(self) -> None:
        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[int, int]] = []
        # Reverse map bytes → token ID, built lazily from merges
        self._bytes_to_id: dict[bytes, int] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text: str, num_merges: int = 500, verbose: bool = True) -> None:
        """
        Train the BPE tokenizer on `text`.

        Parameters
        ----------
        text       : The full training corpus as a Python string.
        num_merges : How many merge operations to perform.  This determines
                     vocabulary size = 256 + num_merges.  Typical values:
                       - 500–1000  for small experiments (fast, small vocab)
                       - 32000      for GPT-2/LLaMA scale
        verbose    : Print progress every 100 merges.

        After training, self.vocab and self.merges are populated and the
        tokenizer is ready to encode/decode.
        """
        # Step 1: Build the base vocabulary — one entry per byte value (0-255).
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []
        self._bytes_to_id = {bytes([i]): i for i in range(256)}

        # Step 2: Convert the training text to byte token sequences.
        # `vocab_tokens` is a list-of-lists of ints (each int 0-255 initially).
        vocab_tokens = _text_to_byte_tokens(text)

        if verbose:
            print(f"[BPE] Starting training: corpus has {len(vocab_tokens)} word-pieces, "
                  f"performing {num_merges} merges...")

        # Step 3: Iteratively find and merge the most frequent pair.
        for merge_idx in range(num_merges):
            # Count all adjacent pairs in the current tokenized corpus.
            pair_counts = _get_pair_counts(vocab_tokens)
            if not pair_counts:
                break  # corpus is fully merged (unlikely with real text)

            # Pick the most frequent pair.  Ties broken by lexicographic order
            # for determinism across runs.
            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            best_count = pair_counts[best_pair]

            if best_count < 2:
                break  # No pair appears more than once — nothing useful to merge

            # Assign this pair a new token ID.
            new_id = 256 + merge_idx

            # Record the merge rule.
            self.merges.append(best_pair)

            # The bytes that this new token represents = bytes of left + bytes of right.
            new_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_id] = new_bytes
            self._bytes_to_id[new_bytes] = new_id

            # Apply the merge to the whole corpus.
            vocab_tokens = _merge_pair(vocab_tokens, best_pair, new_id)

            if verbose and (merge_idx + 1) % 100 == 0:
                print(f"  merge {merge_idx + 1}/{num_merges}: "
                      f"{self.vocab[best_pair[0]]!r} + {self.vocab[best_pair[1]]!r} "
                      f"→ token {new_id} (count={best_count})")

        if verbose:
            print(f"[BPE] Training complete. Vocabulary size: {len(self.vocab)}")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """
        Convert a string to a list of token IDs.

        Algorithm:
          1. Pre-tokenize with the GPT-2 regex (same split used during training).
          2. Convert each piece to a list of byte token IDs (0-255).
          3. Apply learned merge rules in order: for each merge rule (a, b),
             scan the sequence and replace every adjacent (a, b) with the merged ID.
          4. Concatenate all pieces and return the flat list of token IDs.

        This is O(num_merges * sequence_length) per piece — acceptable for
        inference/training data preparation at small scale.
        """
        # Split text into word-pieces using the same regex as training.
        words = regex.findall(GPT2_PAT, text)
        all_ids: list[int] = []

        for word in words:
            # Convert word bytes to initial token IDs (each byte = one token).
            ids: list[int] = list(word.encode("utf-8"))

            # Apply each merge rule in the order they were learned.
            # Earlier merges (more frequent pairs) are applied first, then
            # rarer/longer merges — this mirrors the training order exactly.
            for (a, b), new_id in zip(self.merges, range(256, 256 + len(self.merges))):
                ids = _apply_merge(ids, a, b, new_id)

            all_ids.extend(ids)

        return all_ids

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: list[int]) -> str:
        """
        Convert a list of token IDs back to a string.

        Each token ID maps to a bytes object in self.vocab.  We concatenate
        all those bytes objects and decode the result as UTF-8.

        This is guaranteed lossless: decode(encode(text)) == text for any
        valid UTF-8 string, because:
          - Every token ultimately decomposes to a set of raw byte values.
          - We concatenate the raw bytes and call .decode("utf-8").
          - Since the original text was valid UTF-8, the round-trip is exact.
        """
        byte_pieces = [self.vocab[i] for i in ids]
        raw_bytes = b"".join(byte_pieces)
        return raw_bytes.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Vocabulary size property
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary (256 base + num_merges)."""
        return len(self.vocab)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the tokenizer (vocab + merges) to a JSON file.

        The vocab maps int token IDs to base64-encoded bytes (since raw bytes
        aren't JSON-serializable).  We use a simple hex string instead, which
        is human-readable.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            # Store vocab as {"id": "hex_bytes"} — easy to inspect by hand
            "vocab": {str(k): v.hex() for k, v in self.vocab.items()},
            # Store merges as [[a, b], ...]
            "merges": [[a, b] for a, b in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[BPE] Tokenizer saved to {path}")

    def load(self, path: str) -> None:
        """Load a previously saved tokenizer from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = {int(k): bytes.fromhex(v) for k, v in data["vocab"].items()}
        self.merges = [tuple(pair) for pair in data["merges"]]
        self._bytes_to_id = {v: k for k, v in self.vocab.items()}
        print(f"[BPE] Tokenizer loaded from {path}  (vocab size: {self.vocab_size})")


# ---------------------------------------------------------------------------
# Internal helper used by encode()
# ---------------------------------------------------------------------------

def _apply_merge(ids: list[int], a: int, b: int, new_id: int) -> list[int]:
    """
    Apply a single merge rule (a, b) → new_id to a token ID list.
    Identical logic to _merge_pair but operates on a single sequence.
    """
    result: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result
