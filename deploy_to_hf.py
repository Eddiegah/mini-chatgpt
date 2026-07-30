"""
deploy_to_hf.py — One-shot deploy to Hugging Face Spaces.

Run this AFTER training is complete (best_model.pt must exist).

Usage:
    python deploy_to_hf.py --token YOUR_HF_TOKEN

Get your token at: https://huggingface.co/settings/tokens
Create one with WRITE access.

This script will:
  1. Authenticate with Hugging Face
  2. Create the Space 'minigpt-shakespeare' under your account (if it doesn't exist)
  3. Upload all required files (app, src, tokenizer, model checkpoint)
  4. Print the live public URL
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="Your Hugging Face write token")
    parser.add_argument("--repo-name", default="minigpt-shakespeare", help="Space name (default: minigpt-shakespeare)")
    args = parser.parse_args()

    token = args.token
    repo_name = args.repo_name

    try:
        from huggingface_hub import HfApi, SpaceStage
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=token)

    # Get username
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"Logged in as: {username}")
    except Exception as e:
        print(f"Login failed. Check your token.\nError: {e}")
        sys.exit(1)

    repo_id = f"{username}/{repo_name}"

    # Check required files
    required = [
        "results/checkpoints/best_model.pt",
        "data/tokenizer.json",
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print(f"\nMissing files (run training first):")
        for f in missing:
            print(f"  {f}")
        print("\nRun: python src/train.py")
        sys.exit(1)

    print(f"\nCreating/updating Space: {repo_id}")

    # Create the Space if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            exist_ok=True,
        )
        print(f"Space ready at: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"Error creating Space: {e}")
        sys.exit(1)

    # Files to upload: (local_path, path_in_repo)
    files_to_upload = [
        # App entry point
        ("app.py", "app.py"),
        # Space README (with HF front-matter)
        ("README_SPACES.md", "README.md"),
        # Requirements (space-specific, no matplotlib/tqdm)
        ("space_requirements.txt", "requirements.txt"),
        # Source code
        ("src/__init__.py", "src/__init__.py"),
        ("src/tokenizer.py", "src/tokenizer.py"),
        ("src/embeddings.py", "src/embeddings.py"),
        ("src/attention.py", "src/attention.py"),
        ("src/transformer_block.py", "src/transformer_block.py"),
        ("src/model.py", "src/model.py"),
        ("src/generate.py", "src/generate.py"),
        # Trained artifacts
        ("data/tokenizer.json", "data/tokenizer.json"),
        ("results/checkpoints/best_model.pt", "results/checkpoints/best_model.pt"),
    ]

    print(f"\nUploading {len(files_to_upload)} files...")

    for local_path, repo_path in files_to_upload:
        if not os.path.exists(local_path):
            print(f"  SKIP (not found): {local_path}")
            continue
        size_mb = os.path.getsize(local_path) / 1024 / 1024
        print(f"  Uploading {local_path} ({size_mb:.1f} MB)...", end=" ", flush=True)
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="space",
            )
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")
            sys.exit(1)

    print(f"""
{'='*60}
  Deployment complete!
{'='*60}
  Live URL: https://huggingface.co/spaces/{repo_id}

  The Space is building now (takes ~2-3 min on first deploy).
  Open the URL above — once the build finishes you'll see
  the Gradio interface.
{'='*60}
""")


if __name__ == "__main__":
    main()
