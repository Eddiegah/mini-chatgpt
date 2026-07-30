"""
Evaluation — Perplexity on a held-out validation set.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What is Perplexity?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Perplexity (PPL) measures how "surprised" the model is by a held-out text.

Formally:

    PPL = exp(H)     where H = average cross-entropy loss (in nats)

Concretely:
  - Given N tokens in the validation set, the model predicts each token
    given all previous tokens.
  - The cross-entropy loss for each prediction is -log p(true_next_token).
  - Average this over all N tokens → H (bits per token if log base 2,
    nats per token if natural log — PyTorch uses natural log).
  - PPL = exp(H)

Interpreting perplexity:
  - PPL = 1      → perfect model (assigns probability 1 to every correct token)
  - PPL = V      → random model (assigns uniform probability 1/V to all V tokens)
                   (For vocab_size=756, random PPL ≈ 756)
  - PPL = 50-100 → reasonable range for a SMALL model on a focused corpus
  - GPT-2 (117M) achieves PPL ≈ 35 on WikiText-103
  - GPT-3 (175B) achieves PPL ≈ 20 on Penn Treebank
  - Our MiniGPT (~5M params) on Shakespeare → expect PPL 40-150 after good training
    (anything well below the vocab size means the model has learned something real)

Lower perplexity = better model.  But beware: perplexity is corpus-specific —
you can't compare PPL across different test sets or tokenizers.

Usage:
------
    python src/evaluate.py
"""

import os
import math
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.tokenizer import BPETokenizer
from src.model import MiniGPT, GPTConfig
from src.train import TextDataset

CHECKPOINT_PATH = "results/checkpoints/best_model.pt"
TOKENIZER_PATH = "data/tokenizer.json"
DATA_PATH = "data/corpus.txt"
MAX_SEQ_LEN = 256
BATCH_SIZE = 32
VAL_SPLIT = 0.1


def compute_perplexity(model: MiniGPT, loader: DataLoader, device: torch.device) -> float:
    """
    Compute perplexity on a DataLoader.

    PPL = exp(mean cross-entropy loss over all tokens)
    This is the standard, correct calculation — summing token-level losses
    and dividing by the total token count (not batch count, to avoid
    length-weighting artifacts).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for input_ids, target_ids in tqdm(loader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            _, loss = model(input_ids, target_ids)

            # loss is averaged over tokens in the batch.
            # Multiply back by token count to accumulate unnormalized loss.
            num_tokens = target_ids.numel()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    # Compute mean loss per token and exponentiate to get perplexity.
    mean_loss = total_loss / total_tokens
    perplexity = math.exp(mean_loss)
    return perplexity


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Check for checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[Error] Checkpoint not found at {CHECKPOINT_PATH}")
        print("Run 'python src/train.py' first.")
        return

    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_PATH)

    # Load model
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    config: GPTConfig = ckpt["config"]
    model = MiniGPT(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Eval] Loaded model from epoch {ckpt.get('epoch', '?')}")

    # Prepare validation data
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    token_ids = tokenizer.encode(text)
    split_idx = int(len(token_ids) * (1 - VAL_SPLIT))
    val_ids = token_ids[split_idx:]

    val_dataset = TextDataset(val_ids, MAX_SEQ_LEN)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[Eval] Validation tokens: {len(val_ids):,}")

    # Compute perplexity
    ppl = compute_perplexity(model, val_loader, device)

    # ── Report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Evaluation Results")
    print("=" * 60)
    print(f"  Validation Perplexity: {ppl:.2f}")
    print()
    print("  Interpreting perplexity:")
    print(f"    Random baseline (vocab={tokenizer.vocab_size}): {tokenizer.vocab_size:.0f}")
    print(f"    This model:                                     {ppl:.2f}")
    print()

    if ppl < tokenizer.vocab_size * 0.5:
        print("  ✓ The model is well below the random baseline — it has learned")
        print("    meaningful patterns from the training text.")
    if ppl < 100:
        print("  ✓ PPL < 100: good for a small model on a focused corpus.")
    if ppl < 50:
        print("  ✓ PPL < 50: strong performance for this model scale.")

    print()
    print("  Note: PPL is corpus-specific. These numbers are only comparable")
    print("  to other models evaluated on the same dataset with the same tokenizer.")
    print("=" * 60)


if __name__ == "__main__":
    main()
