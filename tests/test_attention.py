"""
Unit tests for the causal self-attention mechanism.

The most critical correctness property of the whole model:
  The causal mask must genuinely prevent each position from "seeing" future tokens.

Tests:
  1. Output shape is correct.
  2. CAUSALITY TEST (most important): Modifying a future token must NOT change
     the output at any earlier position.  This proves the mask actually works.
  3. Attention weights in the upper triangle are exactly zero.
  4. Multi-head split and merge preserves shape.
  5. Gradient flow check — gradients reach the Q, K, V weights.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attention import CausalSelfAttention


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def attention_module():
    """A small causal attention module for testing."""
    return CausalSelfAttention(
        d_model=64,
        num_heads=4,
        max_seq_len=32,
        dropout=0.0,  # No dropout in tests (would randomise outputs)
    )


@pytest.fixture
def sample_input():
    """A deterministic batch of input tensors."""
    torch.manual_seed(42)
    # (batch=2, seq_len=8, d_model=64)
    return torch.randn(2, 8, 64)


# ── Tests ─────────────────────────────────────────────────────────────────

class TestOutputShape:
    def test_output_shape_matches_input(self, attention_module, sample_input):
        """Output shape must equal input shape."""
        out = attention_module(sample_input)
        assert out.shape == sample_input.shape, (
            f"Expected shape {sample_input.shape}, got {out.shape}"
        )

    def test_batch_size_1(self, attention_module):
        """Works with batch size 1."""
        x = torch.randn(1, 5, 64)
        out = attention_module(x)
        assert out.shape == (1, 5, 64)

    def test_seq_len_1(self, attention_module):
        """Works with a single token (seq_len=1)."""
        x = torch.randn(2, 1, 64)
        out = attention_module(x)
        assert out.shape == (2, 1, 64)


class TestCausality:
    """
    THE KEY TEST: verify the causal mask works correctly.

    The test:
        1. Run a forward pass on input X → outputs O.
        2. Create X_modified: identical to X EXCEPT at a FUTURE position.
        3. Run a forward pass on X_modified → outputs O_modified.
        4. Assert that O[:, :t, :] == O_modified[:, :t, :] for all t < modified_pos.

    If the outputs at positions before `modified_pos` change when we modify
    a future token at `modified_pos`, the mask is broken.
    If they DON'T change, the mask is working correctly.

    This is the cleanest possible proof that no "information leakage" occurs
    from future tokens — if there were any path from future tokens to past
    positions (through broken masking, wrong implementation, etc.), this test
    would catch it.
    """

    def test_future_modification_does_not_affect_past_outputs(self, attention_module):
        """
        CAUSALITY: Modifying a future token should not change any past position's output.
        """
        attention_module.eval()  # Disable dropout
        torch.manual_seed(0)

        batch_size = 1
        seq_len = 8
        d_model = 64

        # Original input
        x = torch.randn(batch_size, seq_len, d_model)

        # Get original outputs
        with torch.no_grad():
            out_original = attention_module(x)

        # For each position t, modify all FUTURE positions and verify
        # that positions 0..t-1 are unchanged.
        for t in range(1, seq_len):
            x_modified = x.clone()

            # Randomly modify all tokens at positions t..seq_len-1
            # (the "future" relative to position t-1)
            x_modified[:, t:, :] = torch.randn(batch_size, seq_len - t, d_model)

            with torch.no_grad():
                out_modified = attention_module(x_modified)

            # Positions BEFORE t must be IDENTICAL in both outputs
            # (the modification at t and beyond should be invisible to 0..t-1)
            past_original = out_original[:, :t, :]
            past_modified = out_modified[:, :t, :]

            max_diff = (past_original - past_modified).abs().max().item()

            assert max_diff < 1e-5, (
                f"CAUSALITY VIOLATION at position t={t}: "
                f"modifying future tokens changed outputs at past positions. "
                f"Max difference: {max_diff:.2e}. "
                f"The causal mask is broken."
            )

    def test_position_0_only_attends_to_itself(self, attention_module):
        """
        Position 0 has no past context, so it can only attend to itself.
        Changing any other token should not affect position 0's output.
        """
        attention_module.eval()
        torch.manual_seed(1)

        x = torch.randn(1, 6, 64)
        with torch.no_grad():
            out_original = attention_module(x)

        # Modify positions 1-5 (all future from position 0's perspective)
        x_modified = x.clone()
        x_modified[:, 1:, :] = torch.randn(1, 5, 64)

        with torch.no_grad():
            out_modified = attention_module(x_modified)

        max_diff = (out_original[:, 0, :] - out_modified[:, 0, :]).abs().max().item()
        assert max_diff < 1e-5, (
            f"Position 0's output changed when modifying future tokens. "
            f"Max diff: {max_diff:.2e}"
        )

    def test_last_position_attends_to_all(self, attention_module):
        """
        The last position CAN attend to all previous positions.
        Changing an earlier token SHOULD change the last position's output.
        """
        attention_module.eval()
        torch.manual_seed(2)

        x = torch.randn(1, 6, 64)
        with torch.no_grad():
            out_original = attention_module(x)

        # Modify position 0 (in the past of the last position)
        x_modified = x.clone()
        x_modified[:, 0, :] = torch.randn(1, 64)

        with torch.no_grad():
            out_modified = attention_module(x_modified)

        # Last position's output SHOULD change (it can attend to position 0)
        last_pos_diff = (
            out_original[:, -1, :] - out_modified[:, -1, :]
        ).abs().max().item()

        assert last_pos_diff > 1e-6, (
            "Last position's output did not change when modifying an earlier token. "
            "The attention mechanism may not be working correctly."
        )


class TestAttentionMaskStructure:
    """Verify the causal mask has the correct lower-triangular structure."""

    def test_causal_mask_is_lower_triangular(self, attention_module):
        """
        The registered causal_mask should be lower-triangular (1s on and below
        the diagonal, 0s above).
        """
        mask = attention_module.causal_mask.squeeze()  # (max_seq_len, max_seq_len)
        T = mask.shape[0]

        for i in range(T):
            for j in range(T):
                expected = 1.0 if j <= i else 0.0
                actual = mask[i, j].item()
                assert actual == expected, (
                    f"mask[{i},{j}] = {actual}, expected {expected}. "
                    f"The causal mask is not correctly lower-triangular."
                )

    def test_mask_is_buffer_not_parameter(self, attention_module):
        """
        The causal mask should be a buffer (not a learned parameter).
        Buffers move with .to(device) but are not updated by the optimizer.
        """
        buffer_names = [name for name, _ in attention_module.named_buffers()]
        assert "causal_mask" in buffer_names, (
            "causal_mask should be a registered buffer, not a parameter"
        )
        param_names = [name for name, _ in attention_module.named_parameters()]
        assert "causal_mask" not in param_names, (
            "causal_mask should NOT be a learnable parameter"
        )


class TestGradientFlow:
    """Verify that gradients flow back through the attention to Q, K, V weights."""

    def test_gradients_reach_qkv_weights(self, attention_module, sample_input):
        """
        After a forward + backward pass, the W_Q, W_K, W_V gradients must be non-None
        and non-zero (i.e. the backward pass propagated through attention correctly).
        """
        x = sample_input.requires_grad_(True)
        out = attention_module(x)
        loss = out.sum()
        loss.backward()

        for name in ["W_Q.weight", "W_K.weight", "W_V.weight", "W_O.weight"]:
            param = dict(attention_module.named_parameters())[name]
            assert param.grad is not None, f"Gradient is None for {name}"
            assert param.grad.abs().sum() > 0, f"Gradient is all zeros for {name}"


class TestMultiHeadSplit:
    """Verify multi-head behavior."""

    def test_different_head_counts_same_dmodel(self):
        """Different num_heads values should all produce correct output shapes."""
        for num_heads in [1, 2, 4, 8]:
            model = CausalSelfAttention(
                d_model=64, num_heads=num_heads, max_seq_len=32, dropout=0.0
            )
            x = torch.randn(1, 10, 64)
            out = model(x)
            assert out.shape == (1, 10, 64), (
                f"num_heads={num_heads}: expected shape (1,10,64), got {out.shape}"
            )

    def test_invalid_head_count_raises(self):
        """d_model must be divisible by num_heads."""
        with pytest.raises(AssertionError):
            CausalSelfAttention(d_model=64, num_heads=7, max_seq_len=32)
