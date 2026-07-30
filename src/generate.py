"""
Text Generation — autoregressive sampling from MiniGPT.

Usage:
------
    # Interactive CLI
    python src/generate.py

    # Direct generation with a prompt
    python src/generate.py --prompt "To be or not to be" --max_tokens 200

    # Greedy decoding
    python src/generate.py --prompt "HAMLET:" --max_tokens 100 --greedy

    # High-temperature sampling (more random/creative)
    python src/generate.py --prompt "KING:" --temperature 1.4 --top_k 50

Sampling Strategies
--------------------
Greedy:
    Always picks the single most likely next token.
    Pros: deterministic, often grammatically correct.
    Cons: tends to repeat itself, lacks variety and creativity.

Temperature sampling:
    Divides logits by T before softmax.
    T < 1.0: more peaked distribution → more confident, more repetitive.
    T = 1.0: unchanged distribution → standard random sampling.
    T > 1.0: flatter distribution → more varied/surprising, can get incoherent.

Top-k + temperature:
    Restrict sampling to the k most probable tokens, then apply temperature.
    Prevents sampling very unlikely tokens while keeping diversity.
    Typical values: top_k=40–50, temperature=0.8–1.0.
"""

import argparse
import os
import torch
from src.tokenizer import BPETokenizer
from src.model import MiniGPT, GPTConfig

CHECKPOINT_PATH = "results/checkpoints/best_model.pt"
TOKENIZER_PATH = "data/tokenizer.json"


def load_model_and_tokenizer(checkpoint_path: str, tokenizer_path: str, device: torch.device):
    """Load the trained model and tokenizer from disk."""
    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Reconstruct config from checkpoint
    config: GPTConfig = ckpt["config"]

    # Build and load model
    model = MiniGPT(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[Generate] Loaded model from epoch {ckpt.get('epoch', '?')} "
          f"(val_loss={ckpt.get('val_loss', '?'):.4f})")
    return model, tokenizer


def generate_text(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int | None = 50,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Generate text given a prompt string.

    Parameters
    ----------
    prompt         : str   — the starting text
    max_new_tokens : int   — how many tokens to generate
    temperature    : float — sampling temperature (0 = greedy)
    top_k          : int   — restrict to top-k tokens (None = no restriction)
    """
    # Encode prompt to token IDs
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        input_ids = [0]  # fallback

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    # Decode full output (prompt + generated)
    full_ids = output_ids[0].tolist()
    return tokenizer.decode(full_ids)


def main():
    parser = argparse.ArgumentParser(description="Generate text with MiniGPT")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Input prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Number of new tokens to generate (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature (default: 0.9; 0=greedy)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling (default: 50; 0=disabled)")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy decoding (overrides temperature)")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--tokenizer", type=str, default=TOKENIZER_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Check for checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"[Error] Checkpoint not found at {args.checkpoint}")
        print("Please run 'python src/train.py' first to train the model.")
        return

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.tokenizer, device)

    temperature = 0.0 if args.greedy else args.temperature
    top_k = None if (args.greedy or args.top_k == 0) else args.top_k

    if args.prompt:
        # Single prompt mode
        print(f"\n{'='*60}")
        print(f"Prompt: {args.prompt!r}")
        print(f"Temperature: {temperature} | Top-k: {top_k}")
        print(f"{'='*60}\n")
        output = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )
        print(output)
    else:
        # Interactive CLI mode
        print("\n" + "="*60)
        print("  MiniGPT Interactive Text Generation")
        print("  Type a prompt and press Enter. Type 'quit' to exit.")
        print("="*60)
        print(f"  Temperature: {temperature} | Top-k: {top_k} | Max tokens: {args.max_tokens}")
        print()

        while True:
            prompt = input("Prompt> ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                break
            if not prompt:
                continue

            output = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=args.max_tokens,
                temperature=temperature,
                top_k=top_k,
                device=device,
            )
            print("\n" + "-"*60)
            print(output)
            print("-"*60 + "\n")


if __name__ == "__main__":
    main()
