---
title: MiniGPT Shakespeare
emoji: 🎭
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: mit
short_description: GPT-style LLM built from scratch — real BPE + causal attention + training
---

# 🎭 MiniGPT — Shakespeare Language Model

A GPT-style transformer language model built **entirely from scratch** in PyTorch — no pretrained weights, no HuggingFace AutoModel, no `nn.MultiheadAttention`. Just the math.

**Type a prompt. Watch it write like Shakespeare.**

---

## What's under the hood

- **BPE tokenizer** built from scratch — same byte-pair encoding algorithm as GPT-2
- **Causal multi-head self-attention** — manually implemented Q/K/V projections, scaled dot-product, causal mask
- **4-layer decoder-only transformer** — pre-norm GPT architecture, ~5M parameters
- **Trained from zero** on Shakespeare's complete works (~1M tokens, 10 epochs)

---

## Sampling strategies

| Strategy | Behaviour |
|---|---|
| **Greedy** | Always picks the most likely token. Deterministic, coherent, can repeat. |
| **Temperature + Top-k** | Controlled randomness. `T=0.9, k=50` is the sweet spot. |
| **Low temperature** | More confident, tighter style. |
| **High temperature** | More surprising/creative, occasionally incoherent. |

---

## Source code & training

Full source, training instructions, Colab notebook, and test suite:  
**[github.com/Eddiegah/mini-chatgpt](https://github.com/Eddiegah/mini-chatgpt)**

Train your own version in ~10 minutes on a free Colab T4 GPU.
