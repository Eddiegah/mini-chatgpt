"""
MiniGPT — Shakespeare Language Model
Streamlit web interface
"""

import os
import sys
import math
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT_PATH = "results/checkpoints/best_model.pt"
TOKENIZER_PATH  = "data/tokenizer.json"

# ── Page config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MiniGPT — Shakespeare",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,300;0,400;0,600;1,400&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default streamlit chrome ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Main container ── */
.block-container {
    padding: 2rem 3rem 3rem 3rem !important;
    max-width: 1200px;
}

/* ── Hero section ── */
.hero {
    background: linear-gradient(135deg, #0d1f17 0%, #1a2f22 50%, #0d1520 100%);
    border: 1px solid #2D6A4F40;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, #2D6A4F20 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Crimson Pro', Georgia, serif;
    font-size: 3rem;
    font-weight: 600;
    color: #E8EAF0;
    margin: 0;
    line-height: 1.1;
}
.hero-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    color: #8B9BB4;
    margin: 0.75rem 0 0 0;
    font-weight: 300;
    line-height: 1.6;
}
.hero-badge {
    display: inline-block;
    background: #2D6A4F25;
    border: 1px solid #2D6A4F60;
    color: #52B788;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    margin: 1rem 0.25rem 0 0;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Status banner ── */
.status-ok {
    background: #1B2D22;
    border: 1px solid #2D6A4F;
    border-left: 4px solid #52B788;
    border-radius: 8px;
    padding: 0.75rem 1.25rem;
    color: #74C69D;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
    font-family: 'Inter', sans-serif;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #556070;
    margin-bottom: 0.5rem;
    margin-top: 1.5rem;
}

/* ── Output box ── */
.output-box {
    background: #111827;
    border: 1px solid #2D6A4F40;
    border-radius: 12px;
    padding: 1.75rem 2rem;
    font-family: 'Crimson Pro', Georgia, serif;
    font-size: 1.15rem;
    line-height: 1.9;
    color: #D4D8E8;
    white-space: pre-wrap;
    min-height: 200px;
    position: relative;
}
.output-box::before {
    content: '"';
    position: absolute;
    top: 0.5rem;
    left: 1rem;
    font-size: 4rem;
    color: #2D6A4F30;
    font-family: Georgia, serif;
    line-height: 1;
}

/* ── Generate button ── */
.stButton > button {
    background: linear-gradient(135deg, #2D6A4F, #1B4332) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px #2D6A4F30 !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #40916C, #2D6A4F) !important;
    box-shadow: 0 6px 20px #2D6A4F50 !important;
    transform: translateY(-1px) !important;
}

/* ── Inputs ── */
.stTextArea textarea {
    background: #111827 !important;
    border: 1px solid #2D3748 !important;
    border-radius: 10px !important;
    color: #E8EAF0 !important;
    font-family: 'Crimson Pro', Georgia, serif !important;
    font-size: 1.1rem !important;
    line-height: 1.7 !important;
    padding: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: #2D6A4F !important;
    box-shadow: 0 0 0 2px #2D6A4F30 !important;
}

/* ── Radio buttons ── */
.stRadio > div {
    gap: 0.4rem !important;
}
.stRadio label {
    background: #1A1F2E !important;
    border: 1px solid #2D3748 !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    font-size: 0.9rem !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}
.stRadio label:hover {
    border-color: #2D6A4F !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #1E2533 !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.5rem !important;
}

/* ── Sliders ── */
.stSlider > div > div > div > div {
    background: #2D6A4F !important;
}

/* ── Divider ── */
hr {
    border-color: #1E2533 !important;
    margin: 1.5rem 0 !important;
}

/* ── Example chips ── */
.example-chip {
    display: inline-block;
    background: #1A1F2E;
    border: 1px solid #2D3748;
    color: #8B9BB4;
    font-size: 0.82rem;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    margin: 0.25rem;
    cursor: pointer;
    transition: all 0.15s ease;
    font-family: 'Crimson Pro', serif;
    font-style: italic;
}

/* ── Stats row ── */
.stat-card {
    background: #1A1F2E;
    border: 1px solid #2D3748;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.stat-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: #52B788;
    font-family: 'Inter', sans-serif;
}
.stat-label {
    font-size: 0.72rem;
    color: #556070;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    import torch  # lazy import — only happens once, cached after
    from src.tokenizer import BPETokenizer
    from src.model import MiniGPT, GPTConfig

    tok = BPETokenizer()
    tok.load(TOKENIZER_PATH)

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    cfg_raw = ckpt["config"]
    if not isinstance(cfg_raw, dict):
        import dataclasses
        cfg_raw = dataclasses.asdict(cfg_raw)

    cfg = GPTConfig(**cfg_raw)
    mdl = MiniGPT(cfg)
    mdl.load_state_dict(ckpt["model_state_dict"])
    mdl.eval()

    val_loss = ckpt.get("val_loss", None)
    ppl = round(math.exp(val_loss), 1) if val_loss else None
    epoch = ckpt.get("epoch", "?")
    return mdl, tok, ppl, epoch, cfg_raw


def run_generate(model, tokenizer, prompt, max_tokens, temperature, top_k, greedy):
    import torch  # lazy import
    ids = tokenizer.encode(prompt) or [0]
    t   = torch.tensor([ids], dtype=torch.long)
    temp = 0.0 if greedy else temperature
    k    = None if greedy else (top_k if top_k > 0 else None)
    with torch.no_grad():
        out = model.generate(t, max_tokens, temperature=temp, top_k=k)
    return tokenizer.decode(out[0].tolist())


# ── Load ─────────────────────────────────────────────────────────────────

with st.spinner("Loading model..."):
    try:
        model, tokenizer, ppl, epoch, cfg = load_model()
        load_ok = True
    except Exception as e:
        load_ok = False
        load_err = str(e)


# ── Hero ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-title">🎭 MiniGPT</div>
    <div class="hero-subtitle">
        A GPT-style transformer built <strong>entirely from scratch</strong> in PyTorch —
        real BPE tokenizer, causal self-attention implemented manually from the math,
        trained on Shakespeare's complete works.
    </div>
    <span class="hero-badge">From Scratch</span>
    <span class="hero-badge">No Pretrained Weights</span>
    <span class="hero-badge">Causal Attention</span>
    <span class="hero-badge">BPE Tokenizer</span>
</div>
""", unsafe_allow_html=True)

# Status
if load_ok:
    st.markdown(f"""
    <div class="status-ok">
        ✓ Model ready — {tokenizer.vocab_size} tokens · perplexity {ppl} · epoch {epoch}
    </div>
    """, unsafe_allow_html=True)
else:
    st.error(f"Failed to load model: {load_err}")
    st.stop()

# Stats row
c1, c2, c3, c4 = st.columns(4)
stats = [
    (cfg.get("d_model", "?"),      "Embedding dim"),
    (cfg.get("num_layers", "?"),   "Layers"),
    (cfg.get("num_heads", "?"),    "Attn heads"),
    (f"{ppl}",                     "Perplexity"),
]
for col, (val, label) in zip([c1, c2, c3, c4], stats):
    col.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{val}</div>
        <div class="stat-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Sidebar: controls ─────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Generation Settings")
    st.markdown("---")

    strategy = st.radio(
        "Sampling strategy",
        ["Temperature + Top-k", "Greedy", "Low temperature", "High temperature"],
        index=0,
        help="Greedy: always picks most likely token. Temperature: controlled randomness.",
    )

    st.markdown("---")

    max_tokens = st.slider("Max new tokens", 30, 400, 200, 10)

    if strategy == "Temperature + Top-k":
        temperature = st.slider("Temperature", 0.1, 2.0, 0.9, 0.05,
                                help="Higher = more creative / random")
        top_k = st.slider("Top-k", 1, 100, 50,
                          help="Restrict sampling to top-k most likely tokens")
    else:
        temperature = {"Greedy": 0.0, "Low temperature": 0.5, "High temperature": 1.4}[strategy]
        top_k = 50
        st.info(f"Temperature: {temperature}")

    greedy = strategy == "Greedy"

    st.markdown("---")
    st.markdown("**Example prompts**")
    examples = [
        "HAMLET:",
        "To be, or not to be,",
        "KING LEAR:",
        "All the world's a stage,",
        "ROMEO:",
        "MACBETH:",
    ]
    for ex in examples:
        st.code(ex, language=None)

    st.markdown("---")
    st.markdown(
        "📦 [Source Code](https://github.com/Eddiegah/mini-chatgpt)  \n"
        "Built by [Eddie](https://github.com/Eddiegah)"
    )


# ── Main: prompt + output ─────────────────────────────────────────────────

st.markdown('<div class="section-label">Your Prompt</div>', unsafe_allow_html=True)

prompt = st.text_area(
    label="prompt",
    value="HAMLET:",
    height=130,
    label_visibility="collapsed",
    placeholder="Enter a Shakespearean prompt...",
)

generate_btn = st.button("✦ Generate Text", type="primary", use_container_width=True)

st.markdown('<div class="section-label">Generated Text</div>', unsafe_allow_html=True)

output_placeholder = st.empty()

if generate_btn:
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Writing..."):
            try:
                result = run_generate(
                    model, tokenizer, prompt,
                    max_tokens, temperature, top_k, greedy
                )
                output_placeholder.markdown(
                    f'<div class="output-box">{result}</div>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Generation error: {e}")
else:
    output_placeholder.markdown(
        '<div class="output-box" style="color:#3D4A5C; font-style:italic;">'
        'Your generated text will appear here...'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Footer ────────────────────────────────────────────────────────────────

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#3D4A5C; font-size:0.8rem; font-family:'Inter',sans-serif;">
    MiniGPT · Built from scratch · Trained on Shakespeare · 
    <a href="https://github.com/Eddiegah/mini-chatgpt" style="color:#52B788;">GitHub</a>
</div>
""", unsafe_allow_html=True)
