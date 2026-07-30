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
short_description: GPT-style language model trained from scratch on Shakespeare
---

# MiniGPT — Shakespeare Language Model

A GPT-style transformer language model built entirely from scratch in PyTorch.

- Real BPE tokenizer (same algorithm as GPT-2's tokenizer)
- Causal multi-head self-attention, manually implemented
- Trained on Shakespeare's complete works (~1M tokens)
- ~5M parameters, 4 layers, 256-dim embeddings

See the [GitHub repo](https://github.com/yourname/minigpt-scratch) for full source and training code.
