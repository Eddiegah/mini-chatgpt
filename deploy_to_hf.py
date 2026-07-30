"""
deploy_to_hf.py — Deploy to Hugging Face Spaces (ZeroGPU — free tier).

ZeroGPU is Hugging Face's free shared GPU tier.
Free accounts get up to 2 Gradio Spaces on ZeroGPU at no cost.

Usage:
    python deploy_to_hf.py --token YOUR_HF_TOKEN

Get your token: https://huggingface.co/settings/tokens (Write access)
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="HuggingFace write token")
    parser.add_argument("--repo-name", default="minigpt-shakespeare")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Run: pip install huggingface_hub"); sys.exit(1)

    api = HfApi(token=args.token)

    # Verify login
    try:
        user = api.whoami()
        username = user["name"]
        print(f"Logged in as: {username}")
    except Exception as e:
        print(f"Login failed: {e}"); sys.exit(1)

    repo_id = f"{username}/{args.repo_name}"

    # Check required files exist
    required = ["results/checkpoints/best_model.pt", "data/tokenizer.json"]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("Missing files:", missing)
        print("Run: python src/train_demo.py"); sys.exit(1)

    print(f"\nCreating Space: {repo_id} (ZeroGPU — free)")

    # Create Space with ZeroGPU hardware
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            exist_ok=True,
        )
        # Set to ZeroGPU hardware (free)
        try:
            api.request_space_hardware(
                repo_id=repo_id,
                hardware="zero-a10g",   # ZeroGPU — free shared GPU
            )
            print("Hardware set to ZeroGPU (free shared GPU)")
        except Exception:
            print("Note: ZeroGPU hardware request skipped (may already be set or unavailable)")

        print(f"Space: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"Error creating Space: {e}")
        print("\nIf you see a payment error, ZeroGPU may not be available for your account yet.")
        print("Alternative: use 'python app.py' locally, or share via 'gradio share'")
        sys.exit(1)

    # Files to upload
    files = [
        ("app.py", "app.py"),
        ("README_SPACES.md", "README.md"),
        ("space_requirements.txt", "requirements.txt"),
        ("src/__init__.py", "src/__init__.py"),
        ("src/tokenizer.py", "src/tokenizer.py"),
        ("src/embeddings.py", "src/embeddings.py"),
        ("src/attention.py", "src/attention.py"),
        ("src/transformer_block.py", "src/transformer_block.py"),
        ("src/model.py", "src/model.py"),
        ("src/generate.py", "src/generate.py"),
        ("data/tokenizer.json", "data/tokenizer.json"),
        ("results/checkpoints/best_model.pt", "results/checkpoints/best_model.pt"),
    ]

    print(f"\nUploading {len(files)} files...")
    for local, remote in files:
        if not os.path.exists(local):
            print(f"  SKIP: {local}")
            continue
        size = os.path.getsize(local) / 1024 / 1024
        print(f"  {local} ({size:.1f} MB)...", end=" ", flush=True)
        try:
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=remote,
                repo_id=repo_id,
                repo_type="space",
            )
            print("✓")
        except Exception as e:
            print(f"FAILED: {e}"); sys.exit(1)

    print(f"""
{'='*60}
  Deployment complete!
{'='*60}
  Live URL: https://huggingface.co/spaces/{repo_id}

  The Space is building (~2-3 min on first deploy).
  Open the URL above once the build finishes.
{'='*60}
""")


if __name__ == "__main__":
    main()
