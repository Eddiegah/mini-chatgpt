"""
Token and Positional Embeddings.

In a transformer, raw token IDs are just integers — they carry no geometric
meaning by themselves.  We need two things:

1. Token Embedding
   ----------------
   A learnable lookup table of shape (vocab_size, d_model).
   Token ID i → a d_model-dimensional vector.
   This vector starts random and is learned during training so that tokens with
   similar roles/meanings end up close to each other in embedding space.

2. Positional Embedding
   ----------------------
   Attention is permutation-equivariant by default: if you shuffle the input
   tokens, the output attention weights shuffle in exactly the same way.
   The model has *no built-in sense of sequence order*.
   Positional embeddings fix this: for each position 0, 1, …, context_len-1
   we add a learned vector that encodes "this token is at position i".

   We use *learnable* positional embeddings (same as GPT-2) rather than the
   fixed sinusoidal embeddings from the original "Attention Is All You Need"
   paper.  Both work; learned ones often perform slightly better in practice
   for fixed-length contexts.

Final embedding:
   x = token_embed(ids) + pos_embed(positions)
   shape: (batch, seq_len, d_model)

The sum is what gets fed into the first transformer block.
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Learnable token embedding table.

    Parameters
    ----------
    vocab_size : int  — number of tokens in the vocabulary
    d_model    : int  — embedding dimension (must match d_model everywhere)
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        # nn.Embedding is a simple lookup table (just a matrix of shape
        # vocab_size × d_model).  During the forward pass, it indexes into
        # this matrix by token ID — effectively a matrix multiplication with
        # a one-hot vector, but implemented efficiently as an index operation.
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : LongTensor of shape (batch, seq_len) — token IDs

        Returns
        -------
        Tensor of shape (batch, seq_len, d_model)
        """
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    """
    Learnable absolute positional embeddings.

    We allocate a table of shape (max_seq_len, d_model).
    For a sequence of length T, we slice out positions 0 … T-1 and add them
    to the token embeddings.

    Parameters
    ----------
    max_seq_len : int  — maximum sequence length the model supports
    d_model     : int  — embedding dimension
    """

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        # Shape: (1, max_seq_len, d_model)
        # The leading 1 is a batch dimension placeholder so broadcasting works
        # when we add to (batch, seq_len, d_model) token embeddings.
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns positional embedding for positions 0 … seq_len-1.

        Parameters
        ----------
        seq_len : int            — current sequence length
        device  : torch.device  — must match the token embeddings device

        Returns
        -------
        Tensor of shape (1, seq_len, d_model)
        """
        # positions = [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        return self.embedding(positions)  # (1, seq_len, d_model)


class GPTEmbedding(nn.Module):
    """
    Combined token + positional embedding module used at the top of the GPT
    stack.

    Parameters
    ----------
    vocab_size  : int   — tokenizer vocabulary size
    d_model     : int   — embedding dimension
    max_seq_len : int   — maximum context window length
    dropout     : float — dropout rate applied after summing embeddings
                         (helps regularize early training)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embed = TokenEmbedding(vocab_size, d_model)
        self.pos_embed = PositionalEmbedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : LongTensor of shape (batch, seq_len) — token IDs

        Returns
        -------
        Tensor of shape (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        # Token embeddings: each ID → its learned vector
        tok = self.token_embed(x)                       # (batch, seq_len, d_model)
        # Positional embeddings: each position → its learned vector
        pos = self.pos_embed(seq_len, x.device)         # (1, seq_len, d_model)
        # Sum and apply dropout.  Broadcasting handles the batch dimension.
        return self.dropout(tok + pos)                  # (batch, seq_len, d_model)
