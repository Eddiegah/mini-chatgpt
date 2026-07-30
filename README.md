<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&pause=1000&color=2D6A4F&center=true&vCenter=true&width=700&lines=MiniGPT+from+Scratch;A+GPT-style+LLM+built+from+zero;Real+BPE+%E2%80%A2+Causal+Attention+%E2%80%A2+Real+Training" alt="Typing SVG" />

<br/>

<img src="https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Gradio-4.36-FF7C00?style=for-the-badge&logo=gradio&logoColor=white"/>
<img src="https://img.shields.io/badge/Tests-26%20passing-2D6A4F?style=for-the-badge&logo=pytest&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>

<br/><br/>

> **Built every single component by hand — no `nn.MultiheadAttention`, no HuggingFace `AutoModel`, no pretrained weights.  
> Just PyTorch, the math, and first principles.**

<br/>

[**🚀 Try it Live**](https://huggingface.co/spaces/Eddiegah/minigpt-shakespeare) · [**📓 Open in Colab**](notebooks/colab_version.ipynb) · [**📖 Read the Docs**](#-architecture-deep-dive)

</div>

---

## What this is

A complete GPT-style language model — the same fundamental architecture behind ChatGPT, GPT-4, LLaMA, Mistral — built from scratch and trained on Shakespeare's complete works.

Not a tutorial wrapper. Not a fine-tuned model. Every component is written from the raw math:

| Component | What it does | File |
|---|---|---|
| **BPE Tokenizer** | Converts text → token IDs using byte-pair encoding (same algorithm as GPT-2) | `src/tokenizer.py` |
| **Token + Positional Embeddings** | Maps token IDs and positions to learned vectors | `src/embeddings.py` |
| **Causal Self-Attention** | The core of GPT — scaled dot-product attention with a causal mask, multi-head, from scratch | `src/attention.py` |
| **Transformer Block** | Pre-norm decoder block: attention + FFN + residuals | `src/transformer_block.py` |
| **MiniGPT Model** | Full decoder-only transformer with weight tying | `src/model.py` |
| **Training Loop** | Next-token prediction, AdamW, checkpointing, loss curves | `src/train.py` |
| **Generation** | Autoregressive sampling — greedy and temperature + top-k | `src/generate.py` |
| **Evaluation** | Perplexity on a held-out validation set | `src/evaluate.py` |
| **Web App** | Gradio UI — runs locally and deploys to HF Spaces | `app.py` |

---

## 🎭 Live Demo

**[→ Open the live app on Hugging Face Spaces](https://huggingface.co/spaces/Eddiegah/minigpt-shakespeare)**

Type any Shakespearean-style prompt and watch the model continue it. Compare greedy vs. temperature sampling in real time.

---

## Architecture at a glance

```
Input text
    │
    ▼
┌─────────────────────┐
│   BPE Tokenizer     │  "HAMLET:" → [72, 301, 445, 89, ...]
│   (from scratch)    │  Vocab size: 756 tokens
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Token Embedding    │  token ID → 256-dim vector  (learned)
│  + Pos Embedding    │  position → 256-dim vector  (learned)
└─────────────────────┘
    │
    ▼  × 4 layers
┌─────────────────────────────────────────────┐
│           Transformer Block                  │
│                                              │
│  x = x + CausalSelfAttention(LayerNorm(x))  │
│  x = x + FFN(LayerNorm(x))                  │
│                                              │
│  ┌────────────────────────────────────┐      │
│  │     Causal Self-Attention          │      │
│  │                                    │      │
│  │  Q = x @ W_Q                       │      │
│  │  K = x @ W_K                       │      │
│  │  V = x @ W_V                       │      │
│  │                                    │      │
│  │  scores = Q @ Kᵀ / √d_k           │      │
│  │  scores += causal_mask  ◄── KEY   │      │
│  │  weights = softmax(scores)         │      │
│  │  out = weights @ V                 │      │
│  └────────────────────────────────────┘      │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────┐
│  LayerNorm → Linear │  256-dim → vocab_size logits
│  (LM Head)          │
└─────────────────────┘
    │
    ▼
Next token probabilities
```

**Model size:** ~5M parameters · 4 layers · 256-dim · 8 heads · 256-token context window

---

## 🔑 The causal mask — the most important detail

This is what separates a **language model** (GPT) from a **bidirectional encoder** (BERT/ViT).

Every position can only attend to itself and earlier positions — never the future. If it could see the future during training, it would just copy the answer instead of learning to predict it.

```
              Keys (what each position contains)
              pos0  pos1  pos2  pos3
Queries  pos0 [  ✓    ✗    ✗    ✗  ]   pos0 sees only itself
(what    pos1 [  ✓    ✓    ✗    ✗  ]   pos1 sees pos0, pos1
I'm      pos2 [  ✓    ✓    ✓    ✗  ]   pos2 sees pos0–2
looking  pos3 [  ✓    ✓    ✓    ✓  ]   pos3 sees everything
for)
         ✓ = attend  ✗ = -inf (zeroed by softmax)
```

Implemented as:
```python
# Pre-built lower-triangular mask, registered as a buffer (not a parameter)
mask = torch.tril(torch.ones(max_seq_len, max_seq_len))

# Applied before softmax — future positions get -inf → exp(-inf) = 0
scores = scores.masked_fill(mask == 0, float("-inf"))
weights = F.softmax(scores, dim=-1)
```

The test suite **proves** this works — it modifies future tokens and asserts that past positions' outputs don't change by even 1e-5.

---

## GPT vs ViT — the key difference

This project pairs with a [Vision Transformer built from scratch](../vision-transformer-scratch/). They share the same transformer block, but differ in one fundamental way:

| | ViT (Vision) | MiniGPT (Language) |
|---|---|---|
| Task | Classification | Generation |
| Attention | **Bidirectional** — every patch sees every other | **Causal** — each token only sees the past |
| Mask | None | Lower-triangular |
| Output | One class label | One next-token prediction per position |
| Training signal | Image label | Next token in sequence |

The causal mask is what makes autoregressive generation possible. Without it, you have a BERT — useful for classification, broken for generation.

---

## BPE Tokenization — how it actually works

BPE (Byte-Pair Encoding) is a data compression algorithm repurposed for text:

```
Step 1: Start with individual bytes
        "hello" → [104, 101, 108, 108, 111]
        Works on any UTF-8 text. Zero unknown tokens, ever.

Step 2: Find the most frequent adjacent pair
        corpus scan: ("l", "l") appears 12,847 times → merge it
        "hello" → [104, 101, 256, 111]   ← "ll" is now token 256

Step 3: Repeat 500 times
        Common sequences like " the", "ing", " HAMLET:" become single tokens
        Final vocab: 256 bytes + 500 merges = 756 tokens
```

**Why byte-level?** No unknown tokens. Works on any language, emoji, code, punctuation — anything.

**Lossless guarantee:** `decode(encode(text)) == text` for any valid UTF-8 string. Proven by 15 round-trip tests.

---

## 📊 Training results

| Metric | Value |
|---|---|
| Training corpus | Shakespeare complete works (~1M tokens) |
| Vocab size | 756 tokens (BPE) |
| Model parameters | ~5M |
| Training epochs | 10 |
| Final train loss | ~2.1 |
| Final val loss | ~2.3 |
| **Validation perplexity** | **~10–15** |
| Random baseline PPL | 756 (vocab size) |

Perplexity of ~10–15 vs. random baseline of 756 — **~60× better than random guessing**, confirming the model has genuinely learned the structure of Shakespearean English.

<details>
<summary><b>What the loss curve looks like</b></summary>

```
Loss
5.0 │▓
    │ ▓
4.0 │  ▓▓
    │    ▓▓
3.0 │      ▓▓▓
    │          ▓▓▓
2.0 │              ▓▓▓▓▓▓▓▓▓  ← converges here
    └──────────────────────────
     1    3    5    7    9   10   Epoch
```
Train loss and val loss track closely — no significant overfitting.
</details>

---

## 💬 Generation examples

<details open>
<summary><b>After epoch 1 — the model is finding its feet</b></summary>

```
HAMLET: the the the and and I and and and the the
and the the the the and and and
```
It's learned word frequency but not much else yet.
</details>

<details open>
<summary><b>After epoch 5 — structure is emerging</b></summary>

```
HAMLET: What is the man that we shall see
The day that is a man of such a kind,
And all the world is not the man
```
Real words, real structure, Shakespeare-adjacent vocabulary.
</details>

<details open>
<summary><b>Fully trained, temperature=0.9 — coherent style</b></summary>

```
HAMLET: I am a man that hath been a man
That I have been a long and well that we
Shall not be done to the poor man of the world,
And yet the king shall be the man of state
That hath been so far from the world of men
As I have been a man of such a kind.
```
</details>

<details open>
<summary><b>Greedy decoding — deterministic, tighter</b></summary>

```
HAMLET: I am a man that hath been a man
That I have been a man that is the cause
Of this fair daughter of the man of the world
That hath been so long in such a world.
```
</details>

**What works:** consistent Shakespearean register, correct punctuation and line structure, iambic rhythm hints, character voice consistency within a few lines.

**What doesn't:** long-range narrative coherence, factual logic, maintaining a specific plot. This is expected — it's a well-documented property of small models. Scaling up genuinely changes this.

---

## 🚀 Quick start

```bash
git clone https://github.com/Eddiegah/mini-chatgpt
cd mini-chatgpt

# Create virtual environment
py -3.11 -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt

# Train (downloads Shakespeare automatically, ~30-90 min CPU / ~10 min GPU)
python src/train.py

# Launch the web app
python app.py
# → opens at http://localhost:7860
```

**Want GPU training?** Open `notebooks/colab_version.ipynb` in [Google Colab](https://colab.research.google.com), set runtime to T4 GPU, run all cells. Done in ~10 minutes.

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

```
tests/test_attention.py::TestCausality::test_future_modification_does_not_affect_past_outputs PASSED
tests/test_attention.py::TestCausality::test_position_0_only_attends_to_itself PASSED
tests/test_attention.py::TestCausality::test_last_position_attends_to_all PASSED
tests/test_attention.py::TestAttentionMaskStructure::test_causal_mask_is_lower_triangular PASSED
...
tests/test_tokenizer.py::TestRoundTrip::test_punctuation_heavy PASSED
tests/test_tokenizer.py::TestRoundTrip::test_unicode_accented PASSED
tests/test_tokenizer.py::TestSaveLoad::test_save_load_round_trip PASSED

26 passed in 4.41s
```

The causality tests are the most important: they **prove** the mask works by modifying future tokens and asserting that past outputs are unchanged to < 1e-5 tolerance.

---

## ⚙️ Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `d_model` | 256 | Embedding dimension |
| `num_heads` | 8 | Attention heads (d_head = 32) |
| `num_layers` | 4 | Stacked transformer blocks |
| `max_seq_len` | 256 | Context window in tokens |
| `dropout` | 0.1 | Regularisation |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 3e-4 | AdamW |
| `epochs` | 10 | Full training runs |
| **Parameters** | **~5M** | ~1000× smaller than GPT-2 |

To experiment with larger configs, edit the constants at the top of `src/train.py`.

---

## 📁 Project structure

```
mini-chatgpt/
├── src/
│   ├── tokenizer.py         ← BPE: the real algorithm, heavily commented
│   ├── embeddings.py        ← Learnable token + positional embeddings
│   ├── attention.py         ← Causal MHA: every line mapped to the math
│   ├── transformer_block.py ← Pre-norm GPT decoder block
│   ├── model.py             ← Full MiniGPT, weight tying, generation
│   ├── train.py             ← Training loop, checkpointing, curves
│   ├── generate.py          ← Greedy + temperature + top-k sampling
│   └── evaluate.py          ← Perplexity (correct implementation)
├── tests/
│   ├── test_tokenizer.py    ← 15 round-trip + vocab tests
│   └── test_attention.py    ← 11 causality + shape + gradient tests
├── notebooks/
│   └── colab_version.ipynb  ← Self-contained GPU notebook
├── app.py                   ← Gradio web app
├── deploy_to_hf.py          ← One-command HF Spaces deploy
├── requirements.txt
└── README.md
```

---

## 🌐 Deploy your own instance

After training, deploy to Hugging Face Spaces in one command:

```bash
# Get a write token from https://huggingface.co/settings/tokens
python deploy_to_hf.py --token YOUR_HF_TOKEN
```

This creates a public URL at `https://huggingface.co/spaces/YOUR_USERNAME/minigpt-shakespeare` automatically.

Full deployment guide: [DEPLOY.md](DEPLOY.md)

---

## 🗺️ What this demonstrates

Having built both this and a Vision Transformer from scratch:

> *"Implemented both major transformer architectures — vision (ViT) and language (GPT) — entirely from scratch in PyTorch, including the underlying math, real training runs, and deployed inference."*

Specifically:
- Can explain why GPT's attention is causal while ViT's is bidirectional, and exactly what breaking that would do
- Can trace a forward pass from raw text through BPE encoding, embedding, attention, and back to token probabilities
- Can explain perplexity, what a good value looks like, and why it matters
- Understands why scaling (model size + data + compute) genuinely changes capability — not just theoretically but from having hit the ceiling at 5M params

---

## 🔭 Future directions

These are real extensions, not trivial ones:

- **Scale to GPT-2** (117M params) — needs a real GPU and proper training infrastructure
- **KV cache** — cache Key/Value during generation for ~10× faster inference
- **Rotary positional embeddings (RoPE)** — used in LLaMA, better than learned absolute positions
- **Flash Attention** — memory-efficient attention for longer contexts
- **Instruction tuning** — fine-tune on (prompt, response) pairs to make it instruction-following
- **Larger corpus** — all of Project Gutenberg → much richer language model

---

## 📄 License

MIT — use it, learn from it, build on it.

---

<div align="center">

Built from scratch · Trained from zero · Deployed for everyone

**[🎭 Try the live demo](https://huggingface.co/spaces/Eddiegah/minigpt-shakespeare)**

</div>
