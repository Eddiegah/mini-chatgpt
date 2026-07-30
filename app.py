"""
MiniGPT — Gradio web interface for Render deployment.
"""

import os
import sys
import math
import torch
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT_PATH = "results/checkpoints/best_model.pt"
TOKENIZER_PATH  = "data/tokenizer.json"

model     = None
tokenizer = None
device    = torch.device("cpu")  # Render free tier is CPU only


# ── Bootstrap: download corpus + train if no checkpoint exists ────────────

def bootstrap():
    """Train a quick model if no checkpoint is present (first deploy)."""
    if os.path.exists(CHECKPOINT_PATH) and os.path.exists(TOKENIZER_PATH):
        return  # already have everything

    print("[Bootstrap] No checkpoint found — running quick training...")
    import urllib.request
    from torch.utils.data import DataLoader
    from src.tokenizer import BPETokenizer
    from src.model import MiniGPT, GPTConfig

    # Download corpus
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/corpus.txt"):
        print("[Bootstrap] Downloading Shakespeare corpus...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            "data/corpus.txt"
        )

    with open("data/corpus.txt", encoding="utf-8") as f:
        text = f.read()[:120_000]  # 120K chars — fast to train

    # Train tokenizer
    tok = BPETokenizer()
    print("[Bootstrap] Training BPE tokenizer...")
    tok.train(text, num_merges=150, verbose=False)
    os.makedirs("data", exist_ok=True)
    tok.save(TOKENIZER_PATH)

    # Tokenize
    ids = tok.encode(text)
    split = int(len(ids) * 0.9)
    train_ids = torch.tensor(ids[:split], dtype=torch.long)
    val_ids   = torch.tensor(ids[split:],  dtype=torch.long)

    SEQ = 64
    BATCH = 128

    def make_batches(data, seq, batch):
        n = (len(data) - seq) // batch * batch
        xs = torch.stack([data[i:i+seq]   for i in range(0, n, 1)][:n//1])
        ys = torch.stack([data[i+1:i+seq+1] for i in range(0, n, 1)][:n//1])
        # simpler: just chunk
        seqs = len(data) - seq
        xs = torch.stack([data[i:i+seq]     for i in range(seqs)])
        ys = torch.stack([data[i+1:i+seq+1] for i in range(seqs)])
        return torch.utils.data.TensorDataset(xs, ys)

    train_ds = make_batches(train_ids, SEQ, BATCH)
    val_ds   = make_batches(val_ids,   SEQ, BATCH)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH)

    # Build model
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        d_model=128, num_heads=4, num_layers=2,
        max_seq_len=SEQ, dropout=0.1,
    )
    m = MiniGPT(cfg).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)

    # Train 3 epochs
    print("[Bootstrap] Training 3 epochs...")
    best_val = float("inf")
    for epoch in range(1, 4):
        m.train()
        for x, y in train_loader:
            opt.zero_grad()
            _, loss = m(x, y)
            loss.backward()
            opt.step()

        m.eval()
        vl = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                _, loss = m(x, y)
                vl += loss.item()
        vl /= len(val_loader)
        print(f"[Bootstrap] Epoch {epoch}/3  val_loss={vl:.4f}")

        if vl < best_val:
            best_val = vl
            os.makedirs("results/checkpoints", exist_ok=True)
            torch.save({
                "epoch": epoch, "model_state_dict": m.state_dict(),
                "val_loss": vl, "config": cfg,
            }, CHECKPOINT_PATH)

    print(f"[Bootstrap] Done. Best val_loss={best_val:.4f}")


# ── Load model ────────────────────────────────────────────────────────────

def load_model():
    global model, tokenizer

    from src.tokenizer import BPETokenizer
    from src.model import MiniGPT

    try:
        tokenizer = BPETokenizer()
        tokenizer.load(TOKENIZER_PATH)

        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        cfg  = ckpt["config"]
        model = MiniGPT(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        val_loss = ckpt.get("val_loss", None)
        ppl = f", perplexity {math.exp(val_loss):.1f}" if val_loss else ""
        return True, f"✓ Model loaded (epoch {ckpt.get('epoch','?')}{ppl}, vocab {tokenizer.vocab_size})"
    except Exception as e:
        return False, f"Error loading model: {e}"


# ── Run bootstrap then load ───────────────────────────────────────────────

bootstrap()
loaded, status_msg = load_model()
print(status_msg)


# ── Generation ────────────────────────────────────────────────────────────

def generate(prompt: str, max_new_tokens: int, temperature: float,
             top_k: int, strategy: str) -> str:
    if model is None or tokenizer is None:
        return "⚠️ Model not loaded."
    if not prompt.strip():
        return "Please enter a prompt."
    try:
        input_ids = tokenizer.encode(prompt) or [0]
        t = torch.tensor([input_ids], dtype=torch.long)

        if strategy == "Greedy":
            temp, k = 0.0, None
        elif strategy == "Low temperature (focused)":
            temp, k = 0.5, top_k
        elif strategy == "High temperature (creative)":
            temp, k = 1.4, top_k
        else:
            temp, k = temperature, top_k

        with torch.no_grad():
            out = model.generate(t, int(max_new_tokens), temperature=temp,
                                 top_k=k if k and k > 0 else None)
        return tokenizer.decode(out[0].tolist())
    except Exception as e:
        return f"Error: {e}"


# ── UI ────────────────────────────────────────────────────────────────────

with gr.Blocks(title="MiniGPT — Shakespeare", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
# 🎭 MiniGPT — Shakespeare Language Model
**A GPT-style transformer built entirely from scratch** — real BPE tokenizer,
causal self-attention implemented manually, trained on Shakespeare's complete works.

📦 [Source Code](https://github.com/Eddiegah/mini-chatgpt) · Built by Eddie
""")

    gr.Textbox(value=status_msg, label="Model Status", interactive=False)

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt", lines=4, value="HAMLET:",
                placeholder="HAMLET:\nTo be, or not to be\nKING LEAR:",
            )
            strategy_radio = gr.Radio(
                choices=["Temperature + Top-k", "Greedy",
                         "Low temperature (focused)", "High temperature (creative)"],
                value="Temperature + Top-k", label="Sampling Strategy",
            )
            with gr.Accordion("Advanced", open=False):
                temp_slider   = gr.Slider(0.1, 2.0, value=0.9, step=0.05, label="Temperature")
                topk_slider   = gr.Slider(1, 200,  value=50,   step=1,    label="Top-k")
                length_slider = gr.Slider(20, 400,  value=200,  step=10,   label="Max new tokens")
            btn = gr.Button("Generate ▶", variant="primary")

        with gr.Column():
            output = gr.Textbox(label="Generated Text", lines=18, show_copy_button=True)

    gr.Examples(
        examples=[
            ["HAMLET:", "Temperature + Top-k", 0.9, 50, 200],
            ["To be, or not to be, that is the question:", "Temperature + Top-k", 0.9, 50, 200],
            ["KING LEAR:", "Greedy", 0.9, 50, 150],
            ["All the world's a stage,", "Low temperature (focused)", 0.5, 40, 200],
        ],
        inputs=[prompt_input, strategy_radio, temp_slider, topk_slider, length_slider],
    )

    btn.click(generate,
              inputs=[prompt_input, length_slider, temp_slider, topk_slider, strategy_radio],
              outputs=output)
    prompt_input.submit(generate,
              inputs=[prompt_input, length_slider, temp_slider, topk_slider, strategy_radio],
              outputs=output)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # share=True creates a public gradio.live link (works everywhere, 72h)
    # On platforms like Render that set PORT, it also binds to 0.0.0.0
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=True,
        show_error=True,
    )
