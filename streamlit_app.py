"""
MiniGPT — Streamlit web interface.

Run locally:
    streamlit run streamlit_app.py
"""

import os
import sys
import math
import torch
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT_PATH = "results/checkpoints/best_model.pt"
TOKENIZER_PATH  = "data/tokenizer.json"

# ── Page config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MiniGPT — Shakespeare",
    page_icon="🎭",
    layout="centered",
)

# ── Load model (cached so it only loads once) ─────────────────────────────

@st.cache_resource
def load_model():
    from src.tokenizer import BPETokenizer
    from src.model import MiniGPT, GPTConfig

    tok = BPETokenizer()
    tok.load(TOKENIZER_PATH)

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    if not isinstance(cfg_dict, dict):
        import dataclasses
        cfg_dict = dataclasses.asdict(cfg_dict)

    cfg = GPTConfig(**cfg_dict)
    mdl = MiniGPT(cfg)
    mdl.load_state_dict(ckpt["model_state_dict"])
    mdl.eval()

    val_loss = ckpt.get("val_loss", None)
    ppl = round(math.exp(val_loss), 1) if val_loss else None
    return mdl, tok, ppl


def generate_text(model, tokenizer, prompt, max_tokens, temperature, top_k, greedy):
    ids = tokenizer.encode(prompt) or [0]
    t = torch.tensor([ids], dtype=torch.long)
    temp = 0.0 if greedy else temperature
    k = None if greedy else top_k
    with torch.no_grad():
        out = model.generate(t, max_tokens, temperature=temp, top_k=k)
    return tokenizer.decode(out[0].tolist())


# ── UI ────────────────────────────────────────────────────────────────────

st.title("🎭 MiniGPT — Shakespeare Language Model")
st.markdown(
    "A GPT-style transformer **built entirely from scratch** — real BPE tokenizer, "
    "causal self-attention implemented manually, trained on Shakespeare's complete works.  \n"
    "📦 [Source Code](https://github.com/Eddiegah/mini-chatgpt)"
)

# Load model
with st.spinner("Loading model..."):
    try:
        model, tokenizer, ppl = load_model()
        st.success(f"✓ Model loaded — vocab {tokenizer.vocab_size} tokens"
                   + (f", perplexity {ppl}" if ppl else ""))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

st.divider()

# Controls
col1, col2 = st.columns([2, 1])

with col1:
    prompt = st.text_area(
        "Prompt",
        value="HAMLET:",
        height=120,
        placeholder="HAMLET:\nTo be, or not to be\nKING LEAR:",
    )

with col2:
    strategy = st.radio(
        "Sampling",
        ["Temperature + Top-k", "Greedy", "Low temp", "High temp"],
        index=0,
    )
    max_tokens = st.slider("Max tokens", 20, 400, 200, 10)

with st.expander("Advanced settings"):
    temperature = st.slider("Temperature", 0.1, 2.0, 0.9, 0.05)
    top_k = st.slider("Top-k", 1, 200, 50)

# Map strategy to params
greedy = strategy == "Greedy"
if strategy == "Low temp":
    temperature = 0.5
elif strategy == "High temp":
    temperature = 1.4

# Generate button
if st.button("▶ Generate", type="primary", use_container_width=True):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating..."):
            try:
                output = generate_text(
                    model, tokenizer, prompt,
                    max_tokens, temperature, top_k, greedy
                )
                st.text_area("Generated Text", value=output, height=300)
            except Exception as e:
                st.error(f"Generation error: {e}")

st.divider()

# Example prompts
st.markdown("**Try these prompts:**")
examples = [
    "HAMLET:",
    "To be, or not to be, that is the question:",
    "KING LEAR:",
    "All the world's a stage,",
    "ROMEO:",
]
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    col.code(ex, language=None)

st.caption(
    "Model: 2-layer transformer · 128-dim · 4 heads · 462K params · "
    "trained on Shakespeare · [GitHub](https://github.com/Eddiegah/mini-chatgpt)"
)
