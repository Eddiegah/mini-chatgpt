"""
Training script for MiniGPT.

Trains the model on a text corpus using next-token prediction (standard LM objective).
Logs training/validation loss, saves checkpoints, and plots training curves.

Usage:
------
    python src/train.py

By default, downloads and trains on Shakespeare's complete works (~5MB text,
public domain, well-suited for a small model).
"""

import os
import sys
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Allow running as both `python src/train.py` and `python -m src.train`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import BPETokenizer
from src.model import MiniGPT, GPTConfig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hyperparameters
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Data ─────────────────────────────────────────────────────────────────
DATA_PATH = "data/corpus.txt"          # Training corpus
TOKENIZER_PATH = "data/tokenizer.json"  # Where to save the trained tokenizer
NUM_MERGES = 500                        # BPE vocabulary size = 256 + 500 = 756

# ── Model ────────────────────────────────────────────────────────────────
D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 4
MAX_SEQ_LEN = 256   # context window in tokens
DROPOUT = 0.1

# ── Training ─────────────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 3e-4
VAL_SPLIT = 0.1         # Fraction of data held out for validation
CHECKPOINT_DIR = "results/checkpoints"
PLOT_PATH = "results/training_curves.png"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dataset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TextDataset(Dataset):
    """
    PyTorch Dataset that yields sliding windows of token IDs.

    Given tokenized text [t0, t1, t2, ..., tN], we create training examples:
        input  = [t0, t1, ..., t_{seq_len-1}]
        target = [t1, t2, ..., t_{seq_len}]

    The model learns to predict target from input — i.e. next-token prediction.
    """

    def __init__(self, token_ids: list[int], seq_len: int) -> None:
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

        # Number of valid windows: if data length = N, we can make N - seq_len
        # windows of length seq_len + 1 (input + target).
        self.num_samples = max(0, len(self.data) - seq_len)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        input_ids : LongTensor of shape (seq_len,)
        target_ids: LongTensor of shape (seq_len,)
        """
        chunk = self.data[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training Loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_epoch(model, loader, optimizer, device, scaler=None):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)

    for input_ids, target_ids in pbar:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()

        # Mixed precision training (optional, faster on GPU with Tensor Cores)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                _, loss = model(input_ids, target_ids)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss = model(input_ids, target_ids)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(loader)


def validate(model, loader, device):
    """Evaluate on validation set. Returns average loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for input_ids, target_ids in tqdm(loader, desc="Validation", leave=False):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            _, loss = model(input_ids, target_ids)
            total_loss += loss.item()
    return total_loss / len(loader)


def plot_losses(train_losses, val_losses, save_path):
    """Plot and save training/validation loss curves."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses, label="Val Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"[Train] Loss plot saved to {save_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 70)
    print("  MiniGPT Training")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Step 1: Prepare corpus ──────────────────────────────────────────
    # If corpus doesn't exist, download Shakespeare
    if not os.path.exists(DATA_PATH):
        print(f"[Data] Corpus not found at {DATA_PATH}.")
        print("[Data] Downloading Shakespeare corpus from Project Gutenberg...")
        os.makedirs("data", exist_ok=True)
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, DATA_PATH)
        print(f"[Data] Downloaded to {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"[Data] Loaded corpus: {len(text):,} characters")

    # ── Step 2: Train tokenizer ─────────────────────────────────────────
    tokenizer = BPETokenizer()
    if os.path.exists(TOKENIZER_PATH):
        print(f"[Tokenizer] Loading from {TOKENIZER_PATH}...")
        tokenizer.load(TOKENIZER_PATH)
    else:
        print(f"[Tokenizer] Training BPE with {NUM_MERGES} merges...")
        tokenizer.train(text, num_merges=NUM_MERGES, verbose=True)
        tokenizer.save(TOKENIZER_PATH)

    # ── Step 3: Tokenize and split ──────────────────────────────────────
    print("[Data] Tokenizing corpus...")
    token_ids = tokenizer.encode(text)
    print(f"[Data] Total tokens: {len(token_ids):,}")

    # Train/val split
    split_idx = int(len(token_ids) * (1 - VAL_SPLIT))
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]
    print(f"[Data] Train: {len(train_ids):,} tokens | Val: {len(val_ids):,} tokens")

    train_dataset = TextDataset(train_ids, MAX_SEQ_LEN)
    val_dataset = TextDataset(val_ids, MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[Data] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Step 4: Build model ─────────────────────────────────────────────
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
    )
    model = MiniGPT(config).to(device)
    print(f"[Model] Total parameters: {model.num_parameters():,}")

    # ── Step 5: Optimizer ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Mixed precision training (optional, GPU only)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Step 6: Training loop ───────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    print("\n" + "=" * 70)
    print("  Starting Training")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        val_loss = validate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Time: {elapsed:.1f}s")

        # Save checkpoint if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "config": config,
            }, checkpoint_path)
            print(f"  → Saved best model (val_loss={val_loss:.4f}) to {checkpoint_path}")

    # ── Step 7: Save final model & plot ─────────────────────────────────
    final_path = os.path.join(CHECKPOINT_DIR, "final_model.pt")
    torch.save({
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, final_path)
    print(f"\n[Train] Final model saved to {final_path}")

    plot_losses(train_losses, val_losses, PLOT_PATH)

    print("\n" + "=" * 70)
    print("  Training Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
