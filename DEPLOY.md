# Deployment Guide

## Option A: Run locally (one command after training)

```
venv\Scripts\activate
python app.py
```

Opens at http://localhost:7860 in your browser automatically.

---

## Option B: Hugging Face Spaces (free public URL, recommended)

This gives you a permanent public link like:
`https://huggingface.co/spaces/YOUR_USERNAME/minigpt`

Anyone can open it and type prompts. No server to manage.

### Prerequisites
- A trained model checkpoint at `results/checkpoints/best_model.pt`
- A trained tokenizer at `data/tokenizer.json`
- A free account at https://huggingface.co

### Step-by-step

**1. Install the HF CLI**
```
pip install huggingface_hub
```

**2. Log in**
```
huggingface-cli login
```
Paste your token from https://huggingface.co/settings/tokens (create one with write access).

**3. Create a new Space on the HF website**
- Go to https://huggingface.co/new-space
- Name it `minigpt` (or whatever you like)
- Select SDK: **Gradio**
- Visibility: Public
- Click Create Space

**4. Clone your new Space locally**
```
git clone https://huggingface.co/spaces/YOUR_USERNAME/minigpt hf-space
cd hf-space
```

**5. Copy the app files into the space**
```
# From your minigpt-scratch project root:
copy app.py hf-space\app.py
copy space_requirements.txt hf-space\requirements.txt
copy README_SPACES.md hf-space\README.md
xcopy /E /I src hf-space\src
copy data\tokenizer.json hf-space\data\tokenizer.json
copy results\checkpoints\best_model.pt hf-space\results\checkpoints\best_model.pt
```

**6. Commit and push**
```
cd hf-space
git add .
git commit -m "Deploy MiniGPT"
git push
```

The Space will build automatically (takes ~2-3 minutes on first deploy).
Your public URL is live at `https://huggingface.co/spaces/YOUR_USERNAME/minigpt`.

### File size note
`best_model.pt` is ~20MB — well within HF's free limits (10GB per repo).
The total Space including PyTorch will be ~900MB, which is fine on the free CPU tier.

### CPU inference speed
On HF Spaces' free CPU tier, generating 200 tokens takes ~10-30 seconds.
This is acceptable for a demo. If you want faster: upgrade to a GPU Space ($~$0.60/hr).

---

## Option C: Temporary public URL for testing (no account needed)

If you just want to share temporarily while the app is running on your machine:

```
# Edit app.py, change the last line to:
demo.launch(share=True)
```

This creates a `*.gradio.live` URL valid for 72 hours. Good for quick demos,
not for permanent hosting.

---

## Option D: Streamlit Community Cloud

Streamlit is an alternative to Gradio. If you prefer Streamlit's style:
1. Replace `app.py` with a Streamlit version (using `st.text_input`, `st.button`, etc.)
2. Push to GitHub
3. Deploy at https://share.streamlit.io — connects to your GitHub repo directly

The Gradio version here is already built and works fine on HF Spaces,
so Streamlit is only worth switching to if you have a specific reason to prefer it.
