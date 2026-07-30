"""
MiniGPT — Gradio web interface.

Runs locally:
    python app.py
    → opens at http://localhost:7860

Deployed on Hugging Face Spaces (ZeroGPU — free):
    Uses @spaces.GPU decorator for shared GPU inference.
    Falls back gracefully to CPU if not on HF Spaces.
"""

import os
import sys
import math
import torch
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── ZeroGPU support (only active on HF Spaces, not on Render) ────────────
try:
    import spaces
    @spaces.GPU
    def _gpu_wrapper(fn):
        return fn
    USE_ZERO_GPU = True
except ImportError:
    USE_ZERO_GPU = False

CHECKPOINT_PATH = "results/checkpoints/best_model.pt"
TOKENIZER_PATH = "data/tokenizer.json"

model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    global model, tokenizer

    from src.tokenizer import BPETokenizer
    from src.model import MiniGPT

    if not os.path.exists(CHECKPOINT_PATH):
        return False, f"No checkpoint found at {CHECKPOINT_PATH}. Train first: python src/train_demo.py"
    if not os.path.exists(TOKENIZER_PATH):
        return False, f"No tokenizer found at {TOKENIZER_PATH}."

    try:
        tokenizer = BPETokenizer()
        tokenizer.load(TOKENIZER_PATH)

        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        config = ckpt["config"]
        model = MiniGPT(config).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        epoch = ckpt.get("epoch", "?")
        val_loss = ckpt.get("val_loss", None)
        ppl = f", perplexity {math.exp(val_loss):.1f}" if val_loss else ""
        return True, f"✓ Model loaded (epoch {epoch}{ppl}, vocab {tokenizer.vocab_size}, device: {device})"
    except Exception as e:
        return False, f"Error: {e}"


def generate(prompt: str, max_new_tokens: int, temperature: float, top_k: int, strategy: str) -> str:
    if model is None or tokenizer is None:
        return "⚠️ Model not loaded."
    if not prompt.strip():
        return "Please enter a prompt."

    try:
        input_ids = tokenizer.encode(prompt) or [0]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        if strategy == "Greedy":
            temp, k = 0.0, None
        elif strategy == "Low temperature (focused)":
            temp, k = 0.5, top_k
        elif strategy == "High temperature (creative)":
            temp, k = 1.4, top_k
        else:
            temp, k = temperature, top_k

        with torch.no_grad():
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=int(max_new_tokens),
                temperature=temp,
                top_k=k if k and k > 0 else None,
            )

        return tokenizer.decode(output_ids[0].tolist())
    except Exception as e:
        return f"Error: {e}"


loaded, status_msg = load_model()
print(status_msg)


with gr.Blocks(
    title="MiniGPT — Shakespeare Language Model",
    theme=gr.themes.Soft(),
    css="""
        .output-text textarea { font-family: Georgia, serif; font-size: 16px; line-height: 1.7; }
        footer { display: none !important; }
    """,
) as demo:

    gr.Markdown("""
# 🎭 MiniGPT — Shakespeare Language Model
**A GPT-style transformer built entirely from scratch** — real BPE tokenizer, causal self-attention implemented manually, trained on Shakespeare's complete works.
No pretrained weights. No `nn.MultiheadAttention`. Just PyTorch and math.

📦 [Source Code & Training](https://github.com/Eddiegah/mini-chatgpt) · Built by [@Eddiegah](https://github.com/Eddiegah)
""")

    gr.Textbox(value=status_msg, label="Model Status", interactive=False)

    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="HAMLET:\nTo be, or not to be\nKING LEAR:",
                lines=4,
                value="HAMLET:",
            )
            strategy_radio = gr.Radio(
                choices=["Temperature + Top-k", "Greedy", "Low temperature (focused)", "High temperature (creative)"],
                value="Temperature + Top-k",
                label="Sampling Strategy",
                info="Greedy = deterministic. Temperature = controlled randomness.",
            )
            with gr.Accordion("Advanced settings", open=False):
                temperature_slider = gr.Slider(0.1, 2.0, value=0.9, step=0.05, label="Temperature")
                topk_slider = gr.Slider(1, 200, value=50, step=1, label="Top-k")
                length_slider = gr.Slider(20, 500, value=200, step=10, label="Max new tokens")
            generate_btn = gr.Button("Generate ▶", variant="primary")

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Generated Text",
                lines=18,
                elem_classes=["output-text"],
                show_copy_button=True,
            )

    gr.Examples(
        examples=[
            ["HAMLET:", "Temperature + Top-k", 0.9, 50, 200],
            ["To be, or not to be, that is the question:", "Temperature + Top-k", 0.9, 50, 200],
            ["KING LEAR:", "Greedy", 0.9, 50, 150],
            ["ROMEO:", "High temperature (creative)", 1.4, 50, 200],
            ["All the world's a stage,", "Low temperature (focused)", 0.5, 40, 200],
        ],
        inputs=[prompt_input, strategy_radio, temperature_slider, topk_slider, length_slider],
        label="Example prompts — click to try",
    )

    gr.Markdown("""
---
**Architecture:** 2-layer decoder-only transformer · 128-dim · 4 heads · 462K parameters · BPE vocab 456 tokens  
**Training:** 3 epochs on 150K chars of Shakespeare · Val loss 3.02 · ~18 min CPU  
**Note:** This is a demo-scale model. For full quality, run `python src/train.py` (10 epochs, full corpus) or use the [Colab notebook](https://github.com/Eddiegah/mini-chatgpt/blob/main/notebooks/colab_version.ipynb).
""")

    generate_btn.click(
        fn=generate,
        inputs=[prompt_input, length_slider, temperature_slider, topk_slider, strategy_radio],
        outputs=output_text,
    )
    prompt_input.submit(
        fn=generate,
        inputs=[prompt_input, length_slider, temperature_slider, topk_slider, strategy_radio],
        outputs=output_text,
    )

if __name__ == "__main__":
    # Render injects PORT=10000 by default.
    # Must bind to 0.0.0.0 so Render's proxy can reach the app.
    port = int(os.environ.get("PORT", 10000))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        root_path=os.environ.get("RENDER_EXTERNAL_URL", ""),
    )
