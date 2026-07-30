"""
MiniGPT — Gradio web interface.

Runs locally:
    python app.py
    → opens at http://localhost:7860

Deployed on Hugging Face Spaces:
    → public URL like https://huggingface.co/spaces/yourname/minigpt

The app lets users type a prompt and see the model's generated continuation,
with controls for temperature and length. Works with the trained checkpoint
from src/train.py (or the Colab notebook).
"""

import os
import math
import torch
import gradio as gr

# ── Try to load the model ────────────────────────────────────────────────
# We do this at startup so the model is ready in memory for fast responses.

CHECKPOINT_PATH = "results/checkpoints/best_model.pt"
TOKENIZER_PATH = "data/tokenizer.json"

model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Load model and tokenizer from disk. Called once at startup."""
    global model, tokenizer

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from src.tokenizer import BPETokenizer
    from src.model import MiniGPT

    if not os.path.exists(CHECKPOINT_PATH):
        return False, f"No checkpoint found at {CHECKPOINT_PATH}. Train the model first with: python src/train.py"

    if not os.path.exists(TOKENIZER_PATH):
        return False, f"No tokenizer found at {TOKENIZER_PATH}. Train the model first."

    try:
        tokenizer = BPETokenizer()
        tokenizer.load(TOKENIZER_PATH)

        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        config = ckpt["config"]
        model = MiniGPT(config).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        epoch = ckpt.get("epoch", "?")
        val_loss = ckpt.get("val_loss", None)
        ppl = math.exp(val_loss) if val_loss else None

        info = f"Model loaded (epoch {epoch}"
        if ppl:
            info += f", val perplexity {ppl:.1f}"
        info += f", vocab {tokenizer.vocab_size}, device: {device})"
        return True, info

    except Exception as e:
        return False, f"Error loading model: {e}"


# ── Generation function ───────────────────────────────────────────────────

def generate(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    strategy: str,
) -> str:
    """
    Called by Gradio when the user clicks Generate.

    Parameters map directly to the UI controls.
    """
    if model is None or tokenizer is None:
        return "⚠️ Model not loaded. Please train the model first (python src/train.py)."

    if not prompt.strip():
        return "Please enter a prompt."

    try:
        # Encode prompt
        input_ids = tokenizer.encode(prompt)
        if not input_ids:
            input_ids = [0]

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        # Determine sampling parameters from strategy selector
        if strategy == "Greedy":
            temp = 0.0
            k = None
        elif strategy == "Low temperature (focused)":
            temp = 0.5
            k = top_k
        elif strategy == "High temperature (creative)":
            temp = 1.4
            k = top_k
        else:  # "Temperature + Top-k" (default)
            temp = temperature
            k = top_k

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=int(max_new_tokens),
                temperature=temp,
                top_k=k if k and k > 0 else None,
            )

        # Decode and return
        full_text = tokenizer.decode(output_ids[0].tolist())
        return full_text

    except Exception as e:
        return f"Generation error: {e}"


# ── Load model at startup ─────────────────────────────────────────────────

loaded, status_msg = load_model()
print(status_msg)


# ── Gradio UI ─────────────────────────────────────────────────────────────

with gr.Blocks(
    title="MiniGPT — Shakespeare Language Model",
    theme=gr.themes.Soft(),
    css="""
        .output-text textarea { font-family: Georgia, serif; font-size: 16px; line-height: 1.7; }
        .status-box { font-size: 12px; color: #666; }
        #generate-btn { background: #2d6a4f; color: white; }
    """,
) as demo:

    gr.Markdown(
        """
        # 🎭 MiniGPT — Shakespeare Language Model
        A GPT-style transformer language model trained from scratch on Shakespeare's complete works.
        Built with a custom BPE tokenizer and manually implemented causal self-attention.
        """
    )

    # Status bar
    with gr.Row():
        status_display = gr.Textbox(
            value=status_msg,
            label="Model Status",
            interactive=False,
            elem_classes=["status-box"],
        )

    with gr.Row():
        # ── Left column: inputs ──────────────────────────────────────────
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="HAMLET:\nTo be, or not to be\nKING LEAR:",
                lines=4,
                value="HAMLET:",
            )

            strategy_radio = gr.Radio(
                choices=[
                    "Temperature + Top-k",
                    "Greedy",
                    "Low temperature (focused)",
                    "High temperature (creative)",
                ],
                value="Temperature + Top-k",
                label="Sampling Strategy",
                info="Greedy = always pick most likely token. Temperature = controlled randomness.",
            )

            with gr.Accordion("Advanced settings", open=False):
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.9,
                    step=0.05,
                    label="Temperature",
                    info="Higher = more random. Only used with Temperature + Top-k strategy.",
                )
                topk_slider = gr.Slider(
                    minimum=1,
                    maximum=200,
                    value=50,
                    step=1,
                    label="Top-k",
                    info="Restrict sampling to top-k most likely tokens.",
                )
                length_slider = gr.Slider(
                    minimum=20,
                    maximum=500,
                    value=200,
                    step=10,
                    label="Max new tokens",
                )

            generate_btn = gr.Button("Generate ▶", variant="primary", elem_id="generate-btn")

        # ── Right column: output ─────────────────────────────────────────
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Generated Text",
                lines=18,
                elem_classes=["output-text"],
                show_copy_button=True,
            )

    # Example prompts
    gr.Examples(
        examples=[
            ["HAMLET:", "Temperature + Top-k", 0.9, 50, 200],
            ["To be, or not to be, that is the question:", "Temperature + Top-k", 0.9, 50, 200],
            ["KING LEAR:", "Greedy", 0.9, 50, 150],
            ["ROMEO:", "High temperature (creative)", 1.4, 50, 200],
            ["All the world's a stage,", "Low temperature (focused)", 0.5, 40, 200],
        ],
        inputs=[prompt_input, strategy_radio, temperature_slider, topk_slider, length_slider],
        label="Example prompts",
    )

    gr.Markdown(
        """
        ---
        **About this model**
        - Architecture: 4-layer decoder-only transformer, 256-dim, 8 attention heads (~5M parameters)
        - Tokenizer: Byte-Pair Encoding (BPE), 756 tokens, trained on the same corpus
        - Training data: Shakespeare's complete works (~1M tokens), 10 epochs
        - Expected output: locally coherent Shakespearean style; limited long-range coherence (expected at this scale)

        **Sampling strategies explained**
        - **Greedy**: always picks the single most likely next token. Deterministic, often repetitive.
        - **Temperature < 1**: more confident predictions, tighter style, less variety.
        - **Temperature > 1**: flatter probability distribution, more surprising choices, can get incoherent.
        - **Top-k**: restricts sampling to the k most probable tokens before applying temperature.
        """
    )

    # Wire up the button
    generate_btn.click(
        fn=generate,
        inputs=[prompt_input, length_slider, temperature_slider, topk_slider, strategy_radio],
        outputs=output_text,
    )

    # Also trigger on Enter in the prompt box
    prompt_input.submit(
        fn=generate,
        inputs=[prompt_input, length_slider, temperature_slider, topk_slider, strategy_radio],
        outputs=output_text,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # accessible on local network too
        share=False,              # set True for a temporary public URL via gradio.live
        show_error=True,
    )
