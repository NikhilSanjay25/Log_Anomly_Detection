"""
HDFS Log Anomaly Detector — Streamlit App
------------------------------------------
Run with:
    streamlit run streamlit_app.py

Requires these files in the same folder:
    transformer_backbone.pth
    rag_random_forest.joblib
    hdfs_faiss_index.index
"""

import os, math
import numpy as np
import torch
import torch.nn as nn
import faiss
import joblib
import google.generativeai as genai
import streamlit as st

# ═══════════════════════════════════════════════════════════
# CONFIG  — edit these two lines
# ═══════════════════════════════════════════════════════════
GEMINI_API_KEY = "AIzaSyAbxZJCXT6MqVQMFv7wlpSqAix39Av54o8"    # ← your key
ARTIFACTS_DIR  = "."                          # folder with the saved files

VOCAB_SIZE = 31
EMB_DIM    = 64
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EVENT2ID = {f"E{i}": i for i in range(1, 30)}
ID2EVENT = {v: k for k, v in EVENT2ID.items()}

# ═══════════════════════════════════════════════════════════
# ARCHITECTURE  (must match training exactly)
# ═══════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb     = nn.Embedding(VOCAB_SIZE, EMB_DIM, padding_idx=0)
        self.pos_enc = PositionalEncoding(EMB_DIM)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=EMB_DIM, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.fc      = nn.Linear(EMB_DIM, 2)

    def extract_features(self, x):
        pad_mask = (x == 0)
        e        = self.pos_enc(self.emb(x))
        out      = self.encoder(e, src_key_padding_mask=pad_mask)
        mask_f   = (~pad_mask).unsqueeze(-1).float()
        return (out * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

    def forward(self, x):
        return self.fc(self.extract_features(x))

# ═══════════════════════════════════════════════════════════
# LOAD ARTIFACTS  (cached so they load only once)
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    # Transformer
    model = TransformerModel().to(DEVICE)
    model.load_state_dict(
        torch.load(
            os.path.join(ARTIFACTS_DIR, "transformer_backbone.pth"),
            map_location=DEVICE,
        )
    )
    model.eval()

    # Random Forest
    rf = joblib.load(os.path.join(ARTIFACTS_DIR, "rag_random_forest.joblib"))

    # FAISS index
    index = faiss.read_index(os.path.join(ARTIFACTS_DIR, "hdfs_faiss_index.index"))

    # Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    gem = genai.GenerativeModel("gemini-2.5-flash")

    return model, rf, index, gem

# ═══════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ═══════════════════════════════════════════════════════════
def parse_log(log_str: str, max_len: int = 60) -> np.ndarray:
    """
    Accepts:  'E2 E5 E11 E9'  or  'E2,E5,E11'  or  '2 5 11'
    Returns int64 array (1, max_len).
    Raises ValueError if no valid tokens found.
    """
    tokens = log_str.replace(",", " ").upper().split()
    ids = []
    for tok in tokens:
        tok = tok.strip()
        if tok.startswith("E") and tok[1:].isdigit():
            ids.append(int(tok[1:]))
        elif tok.isdigit():
            ids.append(int(tok))
    if not ids:
        raise ValueError("No valid event tokens found. Use format: E2 E5 E11 ...")
    ids = ids[-max_len:]
    padded = np.zeros((1, max_len), dtype=np.int64)
    padded[0, :len(ids)] = ids
    return padded

def extract_features(model, seq_array: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x   = torch.tensor(seq_array, dtype=torch.long).to(DEVICE)
        out = model.extract_features(x).cpu().numpy()
    return out  # (1, 64)

def rag_augment(feat: np.ndarray, faiss_index, k: int = 5):
    q = feat.astype(np.float32).copy()
    faiss.normalize_L2(q)
    distances, indices = faiss_index.search(q, k)   # (1, k)
    nb_feats  = np.vstack([faiss_index.reconstruct(int(i)) for i in indices[0]])
    nb_mean   = nb_feats.mean(axis=0, keepdims=True)
    augmented = np.concatenate([feat, nb_mean], axis=1)
    return augmented, indices[0], distances[0]

def decode_seq(ids: np.ndarray) -> str:
    return " → ".join(ID2EVENT.get(int(i), f"E{i}") for i in ids if i != 0)

def get_gemini_explanation(gem, query_ids, prediction, confidence, nb_info):
    label_str  = "ANOMALY" if prediction == 1 else "NORMAL"
    query_text = decode_seq(query_ids[0])
    nb_lines   = "\n".join(
        f"  [{i+1}] Historical sequence #{nb_info[i]}" for i in range(len(nb_info))
    )
    prompt = f"""You are an expert in HDFS distributed system log analysis.

A log sequence has been classified as: {label_str} (confidence: {confidence:.1%})

Query log sequence (event flow):
  {query_text}

Top-5 most similar historical log sequences retrieved for context:
{nb_lines}

Provide:
1. A concise explanation (2-3 sentences) of WHY this sequence was classified as {label_str}.
2. What specific event patterns stand out.
3. Whether the retrieved neighbours support or contradict the prediction.

Be specific about event names. Keep the response under 150 words."""

    try:
        resp = gem.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        return f"⚠️ Gemini unavailable: {e}"

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="HDFS Log Anomaly Detector",
    page_icon="🔍",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

/* Header */
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: 1rem;
    color: #888;
    margin-top: 4px;
    font-weight: 400;
    letter-spacing: 0.5px;
}

/* Result cards */
.result-anomaly {
    background: linear-gradient(135deg, #1a0a0a 0%, #2d1010 100%);
    border: 1.5px solid #ff4444;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 16px 0;
}
.result-normal {
    background: linear-gradient(135deg, #0a1a0e 0%, #102d16 100%);
    border: 1.5px solid #22cc66;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 16px 0;
}
.result-label-anomaly {
    font-size: 2.2rem;
    font-weight: 800;
    color: #ff4444;
    letter-spacing: -0.5px;
}
.result-label-normal {
    font-size: 2.2rem;
    font-weight: 800;
    color: #22cc66;
    letter-spacing: -0.5px;
}
.confidence-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    color: #aaa;
    margin-top: 4px;
}

/* Sequence display */
.seq-box {
    background: #111;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 14px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #e0e0e0;
    word-break: break-all;
    line-height: 1.8;
    margin: 8px 0 16px 0;
}

/* Explanation box */
.explanation-box {
    background: #0d0d14;
    border-left: 3px solid #6c63ff;
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    font-size: 0.95rem;
    color: #d0d0e0;
    line-height: 1.7;
    margin-top: 8px;
}

/* Neighbour pills */
.nb-pill {
    display: inline-block;
    background: #1a1a2e;
    border: 1px solid #333355;
    border-radius: 20px;
    padding: 4px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #9999cc;
    margin: 3px 3px 3px 0;
}

/* Divider */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #555;
    margin: 20px 0 6px 0;
}

/* Sidebar example chips */
.example-chip {
    background: #1e1e1e;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 10px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #bbb;
    margin-bottom: 8px;
    cursor: pointer;
    word-break: break-all;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════
with st.spinner("Loading model artifacts..."):
    try:
        transformer, rf_model, faiss_index, gem = load_artifacts()
        models_ready = True
    except Exception as e:
        models_ready = False
        load_error   = str(e)

# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ System Status")

    if models_ready:
        st.success("All models loaded")
        st.markdown(f"- 🖥️ Device: `{DEVICE}`")
        st.markdown(f"- 🗄️ FAISS vectors: `{faiss_index.ntotal:,}`")
        st.markdown(f"- 🌲 RF estimators: `{rf_model.n_estimators}`")
    else:
        st.error(f"Load error:\n{load_error}")

    st.divider()
    st.markdown("### 📋 Example Logs")
    st.caption("Copy any example into the input box")

    examples = {
        "✅ Normal (typical write)":
            "E2 E5 E5 E5 E11 E9 E11 E9 E11 E9 E26 E26 E26 E23 E23 E23 E21 E21 E21",
        "🚨 Anomaly (abrupt stop)":
            "E22 E5",
        "🚨 Anomaly (unexpected events)":
            "E2 E5 E11 E9 E26 E28 E26 E21 E28 E26 E21",
        "✅ Normal (long sequence)":
            "E2 E5 E5 E11 E9 E11 E9 E26 E26 E11 E9 E26 E23 E23 E21 E21 E21",
    }

    for label, seq in examples.items():
        st.markdown(f"**{label}**")
        st.code(seq, language=None)

    st.divider()
    st.markdown("### 📖 Input Format")
    st.markdown("""
Events separated by spaces or commas:
```
E2 E5 E11 E9 E26
```
Numbers also work:
```
2 5 11 9 26
```
""")

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
st.markdown('<p class="hero-title">🔍 HDFS Log Anomaly Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Transformer + RAG + Gemini · Real-time log sequence analysis</p>', unsafe_allow_html=True)
st.markdown("---")

if not models_ready:
    st.error(f"Cannot run inference — model artifacts failed to load.\n\n`{load_error}`")
    st.info("Make sure `transformer_backbone.pth`, `rag_random_forest.joblib`, and `hdfs_faiss_index.index` are in the same folder as this script.")
    st.stop()

# ── Input ─────────────────────────────────────────────────
col_input, col_info = st.columns([3, 1])

with col_input:
    log_input = st.text_area(
        "Paste your HDFS log sequence here",
        placeholder="E2 E5 E11 E9 E11 E9 E26 E26 E23 E21 E21",
        height=100,
        help="Enter space-separated event IDs, e.g.  E2 E5 E11 E9 E26",
    )

with col_info:
    st.markdown("<br>", unsafe_allow_html=True)
    k_neighbours = st.slider("RAG neighbours (k)", min_value=1, max_value=10, value=5)
    show_explanation = st.toggle("Gemini explanation", value=True)

run_btn = st.button("🔎 Analyse Log", type="primary", use_container_width=True)

# ── Inference ─────────────────────────────────────────────
if run_btn:
    if not log_input.strip():
        st.warning("Please enter a log sequence first.")
        st.stop()

    with st.spinner("Running inference..."):
        try:
            # 1. Parse
            seq_array = parse_log(log_input)
            event_ids = [i for i in seq_array[0] if i != 0]

            # 2. Features
            feat = extract_features(transformer, seq_array)

            # 3. RAG augment
            aug_feat, nb_indices, nb_distances = rag_augment(feat, faiss_index, k=k_neighbours)

            # 4. Classify
            pred       = int(rf_model.predict(aug_feat)[0])
            proba      = rf_model.predict_proba(aug_feat)[0]
            confidence = float(proba[pred])

            # 5. Gemini explanation (optional)
            explanation = ""
            if show_explanation:
                explanation = get_gemini_explanation(
                    gem, seq_array, pred, confidence, nb_indices
                )

        except ValueError as ve:
            st.error(str(ve))
            st.stop()
        except Exception as ex:
            st.error(f"Inference error: {ex}")
            st.stop()

    # ── Results ──────────────────────────────────────────
    is_anomaly = pred == 1
    card_class = "result-anomaly" if is_anomaly else "result-normal"
    lbl_class  = "result-label-anomaly" if is_anomaly else "result-label-normal"
    emoji      = "🚨" if is_anomaly else "✅"
    label_text = "ANOMALY DETECTED" if is_anomaly else "NORMAL SEQUENCE"

    st.markdown(f"""
    <div class="{card_class}">
        <div class="{lbl_class}">{emoji} {label_text}</div>
        <div class="confidence-text">Confidence: {confidence:.1%} &nbsp;|&nbsp;
        Model: Transformer + RAG-RF &nbsp;|&nbsp; Device: {DEVICE}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two-column detail ────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-label">Parsed sequence</div>', unsafe_allow_html=True)
        decoded = decode_seq(seq_array[0])
        st.markdown(f'<div class="seq-box">{decoded}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Class probabilities</div>', unsafe_allow_html=True)
        prob_cols = st.columns(2)
        prob_cols[0].metric("Normal",  f"{proba[0]:.1%}")
        prob_cols[1].metric("Anomaly", f"{proba[1]:.1%}")

        # Mini bar chart
        import pandas as pd
        prob_df = pd.DataFrame({"Class": ["Normal", "Anomaly"], "Probability": [proba[0], proba[1]]})
        st.bar_chart(prob_df.set_index("Class"), color="#ff4444" if is_anomaly else "#22cc66")


    with col_right:
        st.markdown('<div class="section-label">RAG — retrieved neighbours</div>', unsafe_allow_html=True)
        nb_html = "".join(
            f'<span class="nb-pill">#{int(nb_indices[i])} &nbsp; dist={nb_distances[i]:.3f}</span>'
            for i in range(len(nb_indices))
        )
        st.markdown(nb_html, unsafe_allow_html=True)

        if show_explanation and explanation:
            st.markdown('<div class="section-label">Gemini explanation</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="explanation-box">{explanation}</div>', unsafe_allow_html=True)

    # ── Token breakdown ──────────────────────────────────
    with st.expander("🔬 Token-level detail"):
        st.markdown("**Event tokens parsed from your input:**")
        token_cols = st.columns(min(len(event_ids), 10))
        for i, eid in enumerate(event_ids[:10]):
            token_cols[i % len(token_cols)].markdown(
                f"<div style='text-align:center; background:#1a1a2e; border-radius:8px; "
                f"padding:8px; font-family:monospace; font-size:0.9rem; margin:2px;'>"
                f"E{eid}</div>",
                unsafe_allow_html=True,
            )
        if len(event_ids) > 10:
            st.caption(f"... and {len(event_ids)-10} more events")
        st.markdown(f"**Total events:** `{len(event_ids)}` &nbsp; "
                    f"**Feature dim (post-RAG):** `{aug_feat.shape[1]}`")

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.caption("Transformer + Random Forest + FAISS RAG + Gemini 2.5 Flash · HDFS Log Anomaly Detection")