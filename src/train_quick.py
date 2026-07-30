"""
Quick training script — produces a deployable model faster.

Uses fewer BPE merges and a smaller model for a quick end-to-end run.
For best quality, use the full train.py (or the Colab notebook on GPU).

Usage:
    python src/train_quick.py
"""

import os
import sys
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import BPETokenizer
from src.model import MiniGPT, GPTConfig
from src.train import TextDataset, train_epoch, validate, plot_losses

# ── Quick config (faster, still real) ───────────────────────────────────
DATA_PATH = "data/corpus.txt"
TOKENIZER_PATH = "data/tokenizer.json"
NUM_MERGES = 200          # fewer merges = faster tokenizer training
D_MODEL = 128             # smaller model = faster training
NUM_HEADS = 4
NUM_LAYERS = 3
MAX_SEQ_LEN = 128
DROPOUT = 0.1
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 3e-4
VAL_SPLIT = 0.1
CHECKPOINT_DIR = "results/checkpoints"
PLOT_PATH = "results/training_curves.png"


def main():
    print("=" * 60)
    print("  MiniGPT Quick Training (CPU-friendly)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Corpus
    if not os.path.exists(DATA_PATH):
        import urllib.request
        print("Downloading Shakespeare corpus...")
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            DATA_PATH
        )

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Corpus: {len(text):,} characters")

    # Tokenizer
    tokenizer = BPETokenizer()
    if os.path.exists(TOKENIZER_PATH):
        tokenizer.load(TOKENIZER_PATH)
    else:
        print(f"Training BPE tokenizer ({NUM_MERGES} merges)...")
        tokenizer.train(text, num_merges=NUM_MERGES, verbose=True)
        tokenizer.save(TOKENIZER_PATH)

    # Tokenize
    print("Tokenizing corpus...")
    token_ids = tokenizer.encode(text)
    print(f"Total tokens: {len(token_ids):,}")

    split = int(len(token_ids) * (1 - VAL_SPLIT))
    train_ds = TextDataset(token_ids[:split], MAX_SEQ_LEN)
    val_ds   = TextDataset(token_ids[split:], MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # Model
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, max_seq_len=MAX_SEQ_LEN, dropout=DROPOUT,
    )
    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    train_losses, val_losses = [], []
    best_val = float("inf")

    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tl = train_epoch(model, train_loader, optimizer, device)
        vl = validate(model, val_loader, device)
        train_losses.append(tl); val_losses.append(vl)
        print(f"Epoch {epoch}/{EPOCHS} | train={tl:.4f} val={vl:.4f} | {time.time()-t0:.0f}s")

        if vl < best_val:
            best_val = vl
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": tl, "val_loss": vl, "config": config,
            }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            print(f"  → Saved best model (val={vl:.4f})")

    plot_losses(train_losses, val_losses, PLOT_PATH)
    print(f"\nDone! Best val loss: {best_val:.4f}")
    print(f"Model saved to: {CHECKPOINT_DIR}/best_model.pt")


if __name__ == "__main__":
    main()
