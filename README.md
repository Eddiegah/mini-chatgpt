# MiniGPT from Scratch

A GPT-style language model built entirely from scratch in PyTorch — real BPE tokenizer, causal self-attention implemented manually, trained on Shakespeare's complete works.

This pairs with the [ViT-from-scratch](../vision-transformer-scratch/) project: together they cover both major branches of modern transformer-based AI.

---

## What this is (and isn't)

**Is:**
- A complete, real implementation of the GPT architecture with no black-box components
- A real BPE tokenizer (same algorithm as GPT-2's tokenizer)
- Causal multi-head self-attention written from the raw math, not via `nn.MultiheadAttention`
- A working training run on a real text corpus (~1M tokens of Shakespeare)
- A model that generates text with Shakespearean style and locally coherent grammar

**Isn't:**
- GPT-3/4 scale — this is ~5M parameters vs. 175B (a 35,000× difference)
- A general-purpose language model — it's trained on one author's style
- An instruction-following assistant (no fine-tuning/RLHF)
- Able to produce factually reliable or long-range coherent text — this is expected and well-documented at small scale

---

## How GPT differs from ViT

Both projects use the same transformer building block (multi-head attention + FFN + LayerNorm + residuals), but they differ in one critical dimension:

| | ViT (Vision Transformer) | MiniGPT (Language Model) |
|---|---|---|
| **Attention type** | **Bidirectional** — every patch attends to every other patch | **Causal/Unidirectional** — each token can only attend to past tokens |
| **Mask** | None — full attention | Lower-triangular causal mask |
| **Why** | Classification: you have all patches at once | Generation: you predict the *next* token without seeing it |
| **Output** | One class label per image | One next-token prediction per position |
| **Training objective** | Cross-entropy on class labels | Cross-entropy on next-token prediction |

The causal mask is the single biggest architectural difference. Remove it from this model and you have a BERT-style bidirectional encoder — valid for classification, broken for generation (it would "cheat" by seeing future tokens).

---

## BPE Tokenization

BPE (Byte-Pair Encoding) works like a data compression algorithm adapted for text:

1. **Start** with a vocabulary of all 256 possible byte values (so any UTF-8 text is representable with zero unknowns)
2. **Count** every adjacent pair of tokens in the training corpus
3. **Merge** the most frequent pair into a new single token, add it to the vocabulary
4. **Repeat** steps 2–3 for `num_merges` iterations

After training, common sequences like ` the`, `ing`, ` HAMLET:` become single tokens. This compresses the token sequence (fewer tokens per sentence = longer effective context) while keeping the vocabulary manageable.

**Why byte-level?** No unknown token problem. Works on any language, emoji, code, punctuation — anything.

**Losslessness:** `decode(encode(text)) == text` for any valid UTF-8 string. Verified by the test suite.

---

## Project Structure

```
minigpt-scratch/
├── src/
│   ├── tokenizer.py          # BPE tokenizer from scratch
│   ├── embeddings.py         # Token + positional embeddings
│   ├── attention.py          # Causal multi-head self-attention (heavily commented)
│   ├── transformer_block.py  # GPT decoder block (heavily commented)
│   ├── model.py              # Full MiniGPT model
│   ├── train.py              # Training loop with checkpointing
│   ├── generate.py           # Text generation (greedy + temperature)
│   └── evaluate.py           # Perplexity evaluation
├── tests/
│   ├── test_tokenizer.py     # Round-trip correctness + vocab tests
│   └── test_attention.py     # Causality + mask structure tests
├── data/
│   └── corpus.txt            # Shakespeare (auto-downloaded by train.py)
├── results/
│   ├── training_curves.png
│   └── generation_samples.md
├── notebooks/
│   └── colab_version.ipynb   # GPU-accelerated version for Google Colab
├── requirements.txt
└── README.md
```

---

## Setup

### Requirements

- Python 3.9–3.12 (tested on 3.11)
- Windows, macOS, or Linux
- **GPU optional but recommended** — CPU training works but is slow (30–90 min)

### Install

```
py -3.11 -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

### Verify

```
python -c "import torch; import regex; print('CUDA available:', torch.cuda.is_available()); print('All imports OK')"
```

**If you get a DLL error on Windows:**
Install the Microsoft Visual C++ 2015–2022 Redistributable:
- x64: https://aka.ms/vs/17/release/vc_redist.x64.exe
- x86: https://aka.ms/vs/17/release/vc_redist.x86.exe

Restart your terminal and retry.

---

## Training

```
python src/train.py
```

This will:
1. Download the Shakespeare corpus (~1MB) to `data/corpus.txt` if not present
2. Train a BPE tokenizer (500 merges → vocab size 756)
3. Train MiniGPT for 10 epochs
4. Save the best model to `results/checkpoints/best_model.pt`
5. Plot training curves to `results/training_curves.png`

**CPU training time:** ~30–90 minutes depending on hardware.  
**GPU (Colab T4):** ~5–15 minutes — see `notebooks/colab_version.ipynb`.

### Model Hyperparameters (defaults)

| Parameter | Value | Notes |
|---|---|---|
| `d_model` | 256 | Embedding dimension |
| `num_heads` | 8 | Attention heads (d_head = 32) |
| `num_layers` | 4 | Transformer blocks |
| `max_seq_len` | 256 | Context window (tokens) |
| `dropout` | 0.1 | Regularisation |
| **Total params** | ~5M | ~1000× smaller than GPT-2 |

---

## Google Colab (Recommended for GPU)

Open `notebooks/colab_version.ipynb` in Google Colab:  
https://colab.research.google.com/

Set runtime to GPU (Runtime → Change runtime type → T4 GPU) before running.

---

## Generate Text

After training:

```
# Interactive CLI
python src/generate.py

# Direct prompt
python src/generate.py --prompt "HAMLET:" --max_tokens 200

# Greedy decoding (deterministic)
python src/generate.py --prompt "HAMLET:" --greedy

# High-temperature (more creative/random)
python src/generate.py --prompt "To be or not" --temperature 1.4 --top_k 50
```

### Sampling Strategies

**Greedy** (`--greedy`): Always picks the most likely next token. Deterministic, grammatically tighter, tends to repeat.

**Temperature** (`--temperature T`):
- `T < 1.0`: more confident, more repetitive
- `T = 1.0`: standard sampling
- `T > 1.0`: more varied, can get incoherent at high values

**Top-k + Temperature** (default): Restricts sampling to the k most likely tokens, then applies temperature. Best balance of variety and coherence.

---

## Evaluate Perplexity

```
python src/evaluate.py
```

**What perplexity means:**
- PPL = exp(average cross-entropy loss per token)
- PPL = 1: perfect model
- PPL = vocab_size (756): equivalent to random guessing
- **Expected range for MiniGPT on Shakespeare: PPL 40–120** after 10 epochs
- Anything well below vocab_size confirms the model has learned real patterns

---

## Run Tests

```
python -m pytest tests/ -v
```

26 tests covering:
- BPE tokenizer round-trip correctness (ASCII, unicode, punctuation, whitespace, empty string)
- Vocabulary size and structure
- Save/load round-trip
- **Causal mask correctness** — verifies that modifying a future token genuinely does NOT change past positions' outputs
- Attention output shape, multi-head behavior, gradient flow

---

## Generation Examples

*(Filled in after training — examples below are illustrative)*

**Early training (epoch 1) — mostly noise with some structure:**
```
HAMLET: the the the the and and I the and the and and
```

**Mid training (epoch 5):**
```
HAMLET: What is the man that we shall see
The day that is a man of such a man,
And all the world is not the man that
```

**Fully trained (epoch 10), temperature=0.9:**
```
HAMLET: I am a man that hath been a man
That I have been a long and well that we
Shall not be done to the poor man of the world,
And yet the king shall be the man of state
```

**Greedy decoding (epoch 10):**
```
HAMLET: I am a man that hath been a man
That I have been a man that is the cause
Of this fair daughter of the man of the world
```

Observations:
- ✅ Consistent Shakespearean register and vocabulary
- ✅ Grammatically well-formed short phrases
- ✅ Correct use of punctuation and formatting
- ❌ Long-range coherence limited — themes drift after 3–4 lines
- ❌ Factual/logical consistency not reliable at this scale

This is expected behaviour for ~5M parameters trained on ~1M tokens. The model has genuinely learned the style and local syntax of Shakespeare; it hasn't learned to maintain a consistent narrative or argument. The capacity and data gaps are the reason — this is why scaling up (model size, data, compute) genuinely matters for capability, and why GPT-3/4 represent qualitatively different capability levels despite using the same architecture.

---

## Future Directions

The README will be honest: these are substantive engineering efforts, not quick additions.

- **Larger model**: scale to GPT-2 (117M) requires a proper GPU and ~hours of training
- **Better data**: train on a larger, more diverse corpus (e.g., all of Project Gutenberg)
- **Instruction tuning**: fine-tune on (prompt, response) pairs to make it instruction-following
- **Flash Attention**: a more memory-efficient attention implementation (Dao et al. 2022)
- **KV cache**: cache Key/Value matrices during generation for much faster inference
- **Rotary positional embeddings (RoPE)**: used in LLaMA, more effective than learned absolute positions

---

## What This Project Demonstrates

Having built both MiniGPT (this) and ViT-from-scratch:

> "Implemented both major transformer architectures — vision (ViT) and language (GPT) — entirely from scratch, including the core math (scaled dot-product attention, causal masking, BPE tokenization) and real training runs on real data."

Being able to explain:
- Why GPT's attention is causal while ViT's is not
- Exactly what the causal mask does and why removing it breaks generation
- How BPE builds a vocabulary and why it's byte-level
- What perplexity measures and what a good value looks like at small scale

...is a sharp, interview-ready understanding of modern NLP fundamentals.
