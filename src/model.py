"""
MiniGPT — Full Decoder-Only Transformer Language Model.

This module assembles all components into the complete model:

  1. GPTEmbedding  (token + positional, from embeddings.py)
  2. N × TransformerBlock  (causal self-attention + FFN, stacked)
  3. Final LayerNorm  (pre-norm architecture requires a final LN after the stack)
  4. Linear output head  (d_model → vocab_size, producing unnormalised logits)

The output logits[b, t, v] = unnormalised log-probability of token v being
the next token after position t in batch element b.

During training: we apply cross-entropy loss between logits[:, :-1, :] and
targets[:, 1:] (next-token prediction objective).

During inference: we sample or take the argmax of softmax(logits[:, -1, :])
to get the next token, then append and repeat (autoregressive generation).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Default Small-Scale Hyperparameters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  d_model     = 256     embedding dimension
  num_heads   = 8       attention heads (d_head = 32)
  num_layers  = 4       transformer blocks
  max_seq_len = 256     context window (tokens)
  dropout     = 0.1     regularisation

Approximate parameter count at these defaults:
  Embeddings:    vocab × 256 + 256 × 256 ≈ 2M  (vocab ~1256)
  Per block:     ~4 × 256² × 3 (attn) + 256 × 1024 × 2 (FFN) ≈ 800K
  Total (4 blocks): ~5M parameters

This is ~1000× smaller than GPT-2 (117M) and ~300,000× smaller than GPT-3
(175B).  Expectations should be calibrated accordingly: locally coherent
text is achievable; factual reliability and long-range coherence are not.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from src.embeddings import GPTEmbedding
from src.transformer_block import TransformerBlock


@dataclass
class GPTConfig:
    """
    Configuration for MiniGPT.  All hyperparameters in one place.

    Attributes
    ----------
    vocab_size  : size of the BPE vocabulary (set after tokenizer training)
    d_model     : embedding dimension (all layers use this width)
    num_heads   : number of attention heads; must divide d_model evenly
    num_layers  : number of stacked transformer blocks
    max_seq_len : maximum context length in tokens
    dropout     : dropout rate (set to 0.0 for inference-only use)
    """
    vocab_size:  int   = 1000   # overridden after tokenizer.train()
    d_model:     int   = 256
    num_heads:   int   = 8
    num_layers:  int   = 4
    max_seq_len: int   = 256
    dropout:     float = 0.1


class MiniGPT(nn.Module):
    """
    Decoder-only GPT-style language model.

    Parameters
    ----------
    config : GPTConfig — all hyperparameters
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # ── Component stack ────────────────────────────────────────────

        # 1. Embedding layer: token IDs → vectors
        self.embedding = GPTEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # 2. Stack of transformer blocks
        # nn.ModuleList ensures PyTorch tracks these as submodules
        # (for parameter registration, device movement, etc.)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        # 3. Final layer norm (pre-norm architecture requires this after the stack)
        self.ln_final = nn.LayerNorm(config.d_model)

        # 4. Output (language model head): project d_model → vocab_size
        # No softmax here — we return raw logits, and use F.cross_entropy
        # during training (which includes log-softmax internally) or
        # F.softmax during generation.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share weights between the token embedding matrix and
        # the LM head.  This is a standard trick from Press & Wolf (2017) that
        # reduces parameters and often improves performance.  The intuition:
        # the embedding "up-projects" token IDs to vectors, and the head
        # "down-projects" back to logits — using the same matrix for both
        # makes these operations consistent.
        self.lm_head.weight = self.embedding.token_embed.embedding.weight

        # Initialise weights
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        print(f"[MiniGPT] Model initialised. Parameters: {total_params:,}")

    def _init_weights(self) -> None:
        """
        Initialise weights following GPT-2 conventions:
          - Linear layers: N(0, 0.02)
          - Embeddings:    N(0, 0.02)
          - LayerNorm:     weight=1, bias=0
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.

        Parameters
        ----------
        input_ids : LongTensor of shape (batch, seq_len) — token IDs
        targets   : LongTensor of shape (batch, seq_len) — next-token labels
                    If provided, also computes and returns the cross-entropy loss.
                    Pass None during inference (generation).

        Returns
        -------
        logits : Tensor of shape (batch, seq_len, vocab_size)
        loss   : scalar Tensor if targets provided, else None
        """
        # 1. Embeddings: token IDs → continuous vectors
        x = self.embedding(input_ids)  # (B, T, d_model)

        # 2. Pass through all transformer blocks sequentially
        for block in self.blocks:
            x = block(x)               # (B, T, d_model) at each step

        # 3. Final layer norm
        x = self.ln_final(x)           # (B, T, d_model)

        # 4. Project to vocabulary logits
        logits = self.lm_head(x)       # (B, T, vocab_size)

        # 5. Compute loss if targets are provided (training / evaluation mode)
        loss = None
        if targets is not None:
            # Cross-entropy loss for next-token prediction.
            # Reshape:
            #   logits  (B, T, V) → (B*T, V)
            #   targets (B, T)    → (B*T,)
            # F.cross_entropy expects (N, C) and (N,).
            B, T, V = logits.shape
            loss = nn.functional.cross_entropy(
                logits.view(B * T, V),
                targets.view(B * T),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.

        Given a prompt as token IDs, repeatedly:
          1. Run a forward pass on the current sequence.
          2. Take the logits at the LAST position (the next-token prediction).
          3. Sample (or greedy-pick) the next token.
          4. Append it and repeat.

        Parameters
        ----------
        input_ids      : LongTensor (1, seq_len) — prompt token IDs
        max_new_tokens : int — how many new tokens to generate
        temperature    : float — sampling temperature.
                         1.0 = standard sampling.
                         < 1.0 = sharper (more confident, less random).
                         > 1.0 = flatter (more random).
                         0.0 or negative → greedy decoding.
        top_k          : int or None — if set, restrict sampling to the top-k
                         most probable tokens (nucleus-style filtering).

        Returns
        -------
        LongTensor of shape (1, seq_len + max_new_tokens)
        """
        self.eval()  # disable dropout for generation

        for _ in range(max_new_tokens):
            # Crop the context to the model's maximum sequence length.
            # (If the generated sequence grows beyond max_seq_len, we take
            #  only the last max_seq_len tokens as context.)
            ctx = input_ids[:, -self.config.max_seq_len:]

            # Forward pass — we only need logits at the last position.
            logits, _ = self(ctx)                     # (1, T, vocab_size)
            next_logits = logits[:, -1, :]            # (1, vocab_size)

            if temperature <= 0.0:
                # ── Greedy decoding ──────────────────────────────────────
                # Always pick the token with the highest logit.
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # (1, 1)
            else:
                # ── Temperature sampling ─────────────────────────────────
                # Divide logits by temperature before softmax.
                # Low T → scores more peaked → model more confident.
                # High T → scores flatter    → model more random/creative.
                next_logits = next_logits / temperature

                # ── Top-k filtering (optional) ───────────────────────────
                # Zero out all logits except the k largest, then sample.
                # This prevents the model from sampling very unlikely tokens.
                if top_k is not None:
                    # Find the k-th largest value
                    kth_value = next_logits.topk(top_k, dim=-1).values[:, -1, None]
                    # Mask everything below the k-th value
                    next_logits = next_logits.masked_fill(
                        next_logits < kth_value, float("-inf")
                    )

                # Convert to probabilities and sample
                probs = torch.softmax(next_logits, dim=-1)   # (1, vocab_size)
                next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            # Append the new token and continue
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Return the total (or trainable-only) parameter count."""
        params = self.parameters() if not trainable_only else (
            p for p in self.parameters() if p.requires_grad
        )
        return sum(p.numel() for p in params)
