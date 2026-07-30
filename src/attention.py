"""
Causal Multi-Head Self-Attention — from scratch.

This is the core computational primitive of the GPT architecture.  Read
every comment here carefully — this module is where the model "thinks".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1: Scaled Dot-Product Attention
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Given an input matrix X of shape (seq_len, d_model), attention computes:

  Q = X @ W_Q      # Queries  — "what am I looking for?"
  K = X @ W_K      # Keys     — "what do I contain?"
  V = X @ W_V      # Values   — "what will I return if attended to?"

  Attention(Q, K, V) = softmax( Q @ K^T / √d_k ) @ V

where d_k is the per-head dimension.

  • Q @ K^T           — (seq_len, seq_len) matrix of raw similarity scores.
                         Score[i][j] = how much query at position i "matches"
                         key at position j.
  • / √d_k            — Scaling factor.  Without it, when d_k is large the
                         dot products grow large in magnitude, pushing softmax
                         into saturated regions with near-zero gradients.
                         Dividing by √d_k keeps the variance ~1 regardless
                         of dimension.  (Vaswani et al. 2017, §3.2.1)
  • softmax(...)      — Convert raw scores to a probability distribution over
                         positions.  Row i sums to 1.
  • ... @ V           — Weighted sum of value vectors.  Each output position i
                         gets a mixture of all value vectors, weighted by how
                         much query i attended to each key j.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2: THE CAUSAL MASK — the most important correctness property
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

In a language model, we train by asking: "given tokens 0…i, predict token i+1."
This means position i should NEVER be allowed to see positions i+1, i+2, … 
(the future tokens we're trying to predict).  If it could, the model would
simply copy the answer — it would learn nothing.

The causal mask enforces this:

  mask[i][j] = 0     if j <= i  (position j is in the past or present)
  mask[i][j] = -inf  if j > i   (position j is in the future — BLOCKED)

When we add this mask to the raw attention scores before softmax:

  scores_masked[i][j] = score[i][j] + mask[i][j]

For blocked positions: score + (-inf) → softmax output → 0.
Position i genuinely gets zero weight from any future position j > i.

The mask is lower-triangular:

  position:  0  1  2  3
  pos 0:  [  0  -∞ -∞ -∞ ]   # position 0 can only see itself
  pos 1:  [  0   0  -∞ -∞ ]   # position 1 can see 0 and 1
  pos 2:  [  0   0   0  -∞ ]   # position 2 can see 0, 1, 2
  pos 3:  [  0   0   0   0 ]   # position 3 can see all

This is what separates GPT (decoder-only causal model) from BERT/ViT
(encoder/bidirectional models where every position attends to every other).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3: Multi-Head Attention
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Instead of one big attention operation, we split d_model into `num_heads`
smaller heads of size d_head = d_model // num_heads.

Each head has its own W_Q, W_K, W_V projections and computes attention
independently.  The outputs are concatenated and projected back to d_model.

Why?  Each head can specialise in attending to different types of
relationships (syntax, semantics, coreference, etc.) simultaneously.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention, implemented from scratch.

    Parameters
    ----------
    d_model   : int   — total embedding dimension (must be divisible by num_heads)
    num_heads : int   — number of attention heads
    max_seq_len: int  — maximum sequence length (used to pre-build the causal mask)
    dropout   : float — attention weight dropout (regularisation)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads  # dimension per head

        # ── Linear projections ──────────────────────────────────────────
        # We use three separate linear layers (no bias, matching GPT-2 convention).
        # Each maps (batch, seq, d_model) → (batch, seq, d_model).
        # The per-head splits happen *after* the projection (see forward()).
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # Output projection: concatenated heads → d_model
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # Dropout applied to attention weights (before the weighted sum).
        self.attn_dropout = nn.Dropout(dropout)
        # Dropout applied to the output projection.
        self.proj_dropout = nn.Dropout(dropout)

        # ── Causal mask ────────────────────────────────────────────────
        # Pre-compute a (1, 1, max_seq_len, max_seq_len) lower-triangular mask.
        # We register it as a buffer so it moves to the correct device
        # automatically with .to(device) / .cuda(), but is NOT a parameter
        # (it's not learned — it's fixed).
        #
        # torch.tril(torch.ones(T, T)) produces:
        #   [[1, 0, 0, ...],
        #    [1, 1, 0, ...],
        #    [1, 1, 1, ...],  ← lower triangle of 1s
        #    ...]
        # Shape: (1, 1, T, T) so it broadcasts over (batch, heads, T, T).
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, d_model)

        Returns
        -------
        Tensor of shape (batch, seq_len, d_model)

        Step-by-step walkthrough
        ─────────────────────────
        1. Project x to Q, K, V  (all shape: batch × seq × d_model)
        2. Split each into num_heads heads  (batch × heads × seq × d_head)
        3. Compute raw attention scores: Q @ K^T  (batch × heads × seq × seq)
        4. Scale by 1/√d_head
        5. Apply the causal mask (add -inf to upper triangle)
        6. Softmax → attention weights
        7. Dropout on attention weights
        8. Weighted sum with V  → (batch × heads × seq × d_head)
        9. Reshape back to  (batch × seq × d_model)
        10. Final linear projection W_O
        """
        B, T, C = x.shape  # batch, seq_len, d_model

        # ── Step 1: Compute Q, K, V projections ─────────────────────────
        # Each of shape (B, T, d_model)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # ── Step 2: Split d_model into num_heads × d_head ───────────────
        # Reshape: (B, T, d_model) → (B, T, num_heads, d_head)
        # Transpose: → (B, num_heads, T, d_head)
        # This groups the attention computation by head so we can do all
        # heads in one batched matrix multiply.
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
            # result shape: (B, num_heads, T, d_head)

        Q = split_heads(Q)  # (B, H, T, d_head)
        K = split_heads(K)  # (B, H, T, d_head)
        V = split_heads(V)  # (B, H, T, d_head)

        # ── Step 3 & 4: Scaled dot-product attention scores ─────────────
        # Q @ K^T gives raw similarity scores.
        # K.transpose(-2, -1) swaps the last two dims: (B, H, d_head, T)
        # Matmul: (B, H, T, d_head) @ (B, H, d_head, T) = (B, H, T, T)
        #
        # Score[b][h][i][j] = how much query at position i attends to key at j
        scale = math.sqrt(self.d_head)
        scores = (Q @ K.transpose(-2, -1)) / scale  # (B, H, T, T)

        # ── Step 5: Apply the causal mask ───────────────────────────────
        # self.causal_mask shape: (1, 1, max_seq_len, max_seq_len)
        # Slice to current sequence length T: (1, 1, T, T)
        mask = self.causal_mask[:, :, :T, :T]  # (1, 1, T, T)

        # Where mask == 0 (upper triangle = future positions), set score to -inf.
        # After softmax, exp(-inf) = 0 → those positions get zero attention weight.
        # This is the causal mask in action.
        scores = scores.masked_fill(mask == 0, float("-inf"))  # (B, H, T, T)

        # ── Step 6: Softmax over key dimension ──────────────────────────
        # Dim=-1 means softmax over the "key" axis, so each query position's
        # weights sum to 1 across all (visible) key positions.
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, T, T)

        # ── Step 7: Dropout on attention weights ────────────────────────
        # Randomly zero out some attention connections during training.
        attn_weights = self.attn_dropout(attn_weights)

        # ── Step 8: Weighted sum of values ──────────────────────────────
        # (B, H, T, T) @ (B, H, T, d_head) = (B, H, T, d_head)
        # For each query position, we get a d_head-dimensional weighted mixture
        # of all value vectors.
        out = attn_weights @ V  # (B, H, T, d_head)

        # ── Step 9: Merge heads back ─────────────────────────────────────
        # Transpose back: (B, H, T, d_head) → (B, T, H, d_head)
        # contiguous() is required before view() if memory layout changed.
        # view: → (B, T, d_model)   [where d_model = H * d_head]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # ── Step 10: Output projection ───────────────────────────────────
        # Mix information across heads.
        out = self.W_O(out)          # (B, T, d_model)
        out = self.proj_dropout(out)

        return out
