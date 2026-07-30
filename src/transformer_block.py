"""
GPT-style Transformer Decoder Block — from scratch.

A single transformer block combines four components:
  1. Layer Normalization (pre-norm, applied *before* sublayers)
  2. Causal Multi-Head Self-Attention
  3. Layer Normalization (again, before FFN)
  4. Feed-Forward Network (FFN / MLP)

With residual connections wrapping steps 2 and 4.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Architecture Choice: Pre-Norm vs Post-Norm
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The original "Attention Is All You Need" paper used POST-norm:
    x = LayerNorm(x + Sublayer(x))

GPT-2 and most modern models use PRE-norm:
    x = x + Sublayer(LayerNorm(x))

Pre-norm trains more stably, especially with deeper networks and without
a careful learning rate warm-up schedule.  We use pre-norm here.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Residual Connections
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Both the attention and FFN sublayers are wrapped in residual connections:
    x = x + Sublayer(x)   (simplified; the actual input to Sublayer is LN(x))

Residuals allow gradients to flow directly back through the entire network
during backpropagation — without them, deep networks (>~6 layers) suffer
from vanishing gradients and fail to train.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Feed-Forward Network (FFN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The FFN is a simple two-layer MLP applied position-wise (independently to
each position):
    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2

  • Inner dimension: typically 4 × d_model (e.g. d_model=256 → inner=1024).
    This expansion gives the network capacity to compute complex functions of
    individual token representations.
  • Activation: GELU (Gaussian Error Linear Unit) — smoother than ReLU,
    empirically better for transformers (used in GPT-2/3).
  • Applied per-position: the FFN does NOT mix information between positions
    (that's the attention's job).  It processes each position's representation
    independently.
"""

import torch
import torch.nn as nn
from src.attention import CausalSelfAttention


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Architecture: Linear → GELU → Linear → Dropout
    Dimensions:   d_model → 4*d_model → d_model

    Parameters
    ----------
    d_model  : int   — model dimension (input and output)
    dropout  : float — dropout applied after the second linear layer
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        # The inner dimension is 4 × d_model, following the GPT-2 convention.
        inner_dim = 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, inner_dim),   # expand
            nn.GELU(),                        # non-linearity
            nn.Linear(inner_dim, d_model),   # contract back
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, d_model)

        Returns
        -------
        Tensor of shape (batch, seq_len, d_model)

        The FFN is applied identically to every position (it's a simple
        matrix multiply — no interaction between positions here).
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One GPT-style transformer decoder block (pre-norm).

    Data flow:
    ─────────────────────────────────────────────────────────────────────
    x  ──┬──► LayerNorm ──► CausalAttention ──► (+) ──► x'
         │                                        ▲
         └────────────────────────────────────────┘  (residual)

    x' ──┬──► LayerNorm ──► FFN ──► (+) ──► output
         │                           ▲
         └───────────────────────────┘  (residual)
    ─────────────────────────────────────────────────────────────────────

    Parameters
    ----------
    d_model     : int   — embedding dimension
    num_heads   : int   — number of attention heads
    max_seq_len : int   — needed to pre-build causal mask in attention
    dropout     : float — dropout rate throughout
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # LayerNorm before attention sublayer
        self.ln1 = nn.LayerNorm(d_model)
        # Causal multi-head self-attention (from scratch, see attention.py)
        self.attn = CausalSelfAttention(d_model, num_heads, max_seq_len, dropout)

        # LayerNorm before FFN sublayer
        self.ln2 = nn.LayerNorm(d_model)
        # Position-wise feed-forward network
        self.ffn = FeedForwardNetwork(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, d_model)

        Returns
        -------
        Tensor of shape (batch, seq_len, d_model)
        """
        # ── Attention sublayer (pre-norm + residual) ───────────────────
        # Pre-norm: normalize *before* computing attention.
        # Residual: add input x back to the attention output.
        x = x + self.attn(self.ln1(x))

        # ── FFN sublayer (pre-norm + residual) ─────────────────────────
        x = x + self.ffn(self.ln2(x))

        return x
