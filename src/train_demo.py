"""
Demo training script — trains a real model in ~5-10 minutes on CPU.

Uses a small slice of the corpus and a compact model so it completes quickly.
The result is a real trained model that generates recognizable Shakespeare-style text.

For the full-quality model (recommended), use:
  - Google Colab notebook (notebooks/colab_version.ipynb) on T4 GPU — ~10 min
  - python src/train.py locally — ~2-4 hours CPU

Usage:
    python src/train_demo.py
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
from src.train import TextDataset, plot_losses

# ── Demo config ──────────────────────────────────────────────────────────
DATA_PATH       = "data/corpus.txt"
TOKENIZER_PATH  = "data/tokenizer.json"
NUM_MERGES      = 150       # Fast to train, vocab=406
CORPUS_CHARS    = 150_000   # Use first 150K chars (~25% of Shakespeare)
D_MODEL         = 128
NUM_HEADS       = 4
NUM_LAYERS      = 2
MAX_SEQ_LEN     = 64
DROPOUT         = 0.1
BATCH_SIZE      = 128
EPOCHS          = 3
LEARNING_RATE   = 3e-4
VAL_SPLIT       = 0.1
CHECKPOINT_DIR  = "results/checkpoints"
PLOT_PATH       = "results/training_curves.png"


def train_epoch(model, loader, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for x, y in tqdm(loader, desc="  Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        total += loss.item(); n += 1
    return total / n


def validate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="  Val  ", leave=False):
            _, loss = model(x.to(device), y.to(device))
            total += loss.item(); n += 1
    return total / n


def main():
    print("=" * 60)
    print("  MiniGPT Demo Training (~5-10 min on CPU)")
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
        full_text = f.read()

    # Use only a slice for demo speed
    text = full_text[:CORPUS_CHARS]
    print(f"Using {len(text):,} / {len(full_text):,} characters")

    # Tokenizer
    tokenizer = BPETokenizer()
    if os.path.exists(TOKENIZER_PATH):
        print("Loading existing tokenizer...")
        tokenizer.load(TOKENIZER_PATH)
    else:
        print(f"Training BPE tokenizer ({NUM_MERGES} merges)...")
        tokenizer.train(text, num_merges=NUM_MERGES, verbose=True)
        tokenizer.save(TOKENIZER_PATH)

    # Tokenize
    print("Tokenizing...")
    token_ids = tokenizer.encode(text)
    print(f"Tokens: {len(token_ids):,}  Vocab: {tokenizer.vocab_size}")

    split = int(len(token_ids) * (1 - VAL_SPLIT))
    train_ds = TextDataset(token_ids[:split], MAX_SEQ_LEN)
    val_ds   = TextDataset(token_ids[split:], MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Batches — train: {len(train_loader)}  val: {len(val_loader)}")

    # Model
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN, dropout=DROPOUT,
    )
    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    train_losses, val_losses = [], []
    best_val = float("inf")

    print(f"\nTraining {EPOCHS} epochs...\n")
    total_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tl = train_epoch(model, train_loader, optimizer, device)
        vl = validate(model, val_loader, device)
        train_losses.append(tl); val_losses.append(vl)
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{EPOCHS} | train={tl:.4f}  val={vl:.4f} | {elapsed:.0f}s")

        if vl < best_val:
            best_val = vl
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": tl, "val_loss": vl, "config": config,
            }, ckpt_path)
            print(f"  ✓ Saved (val={vl:.4f}) → {ckpt_path}")

        # Quick generation sample after each epoch
        model.eval()
        prompt_ids = tokenizer.encode("HAMLET:")[:5] or [65]
        prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(prompt_t, max_new_tokens=60, temperature=0.9, top_k=40)
        sample = tokenizer.decode(out[0].tolist())
        print(f"  Sample: {sample[:120]!r}\n")

    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time/60:.1f} minutes")
    print(f"Best val loss: {best_val:.4f}")

    # Plot
    plot_losses(train_losses, val_losses, PLOT_PATH)

    print(f"\n✓ Model ready at: {CHECKPOINT_DIR}/best_model.pt")
    print(f"✓ Tokenizer at:   {TOKENIZER_PATH}")
    print(f"\nRun the web app:  python app.py")
    print(f"Generate text:    python src/generate.py")


if __name__ == "__main__":
    main()
