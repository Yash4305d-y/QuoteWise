import os
import pickle
import random
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Page configuration
st.set_page_config(
    page_title="QuoteWise",
    page_icon="✍️",
    layout="centered"
)

# Custom CSS for clean presentation
st.markdown("""
    <style>
        .quote-card {
            background-color: #1e2530;
            border-left: 4px solid #ff4b4b;
            padding: 16px 20px;
            border-radius: 8px;
            margin-bottom: 12px;
            font-size: 1.15rem;
            font-style: italic;
            color: #f1f5f9;
        }
        .meta-tag {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 4px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("✍️ QuoteWise")
st.caption("cuDNN-accelerated LSTM Language Model")

st.warning(
    "⚠️ **Scope Notice:** This model is trained exclusively on philosophical and literary quotes. "
    "Use classic proverb starters (e.g., *'The secret to'*, *'In the middle of'*). "
    "Conversational queries (*'how are u'*) will yield disconnected results."
)

# 1. Cached Asset Loader
@st.cache_resource(show_spinner="Loading LSTM model and tokenizer...")
def load_assets():
    if not os.path.exists("lstm_quote_model.h5"):
        raise FileNotFoundError("Missing lstm_quote_model.h5")
    if not os.path.exists("tokenizer.pkl"):
        raise FileNotFoundError("Missing tokenizer.pkl")
    if not os.path.exists("config.pkl"):
        raise FileNotFoundError("Missing config.pkl")

    model = tf.keras.models.load_model("lstm_quote_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("config.pkl", "rb") as f:
        config = pickle.load(f)

    return model, tokenizer, config.get("max_len", 100)

try:
    model, tokenizer, max_len = load_assets()
    index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# 2. Advanced Sampling (Top-P + Top-K + Repetition Penalty)
def sample_token(preds, temperature=0.38, top_k=15, top_p=0.85, recent_tokens=None, rep_penalty=1.25):
    preds = np.asarray(preds).astype("float64")
    
    # Repetition penalty on recent tokens
    if recent_tokens:
        for t in set(recent_tokens[-4:]):
            if preds[t] > 0:
                preds[t] /= rep_penalty

    # Temperature scaling
    preds = np.log(preds + 1e-8) / max(temperature, 1e-4)
    exp_preds = np.exp(preds)
    probs = exp_preds / np.sum(exp_preds)

    # Top-K Filtering
    if top_k > 0:
        top_k_indices = np.argsort(probs)[-top_k:]
        k_mask = np.zeros_like(probs)
        k_mask[top_k_indices] = probs[top_k_indices]
        probs = k_mask / np.sum(k_mask)

    # Top-P (Nucleus) Filtering
    if 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        
        # Remove tokens outside the nucleus
        remove_indices = cumulative_probs > top_p
        remove_indices[1:] = remove_indices[:-1]
        remove_indices[0] = False
        
        probs[sorted_indices[remove_indices]] = 0.0
        probs = probs / np.sum(probs)

    return np.argmax(np.random.multinomial(1, probs, 1))

def generate_quote(prompt, length):
    generated = prompt.strip()
    history_indices = []
    
    for _ in range(length):
        tokens = tokenizer.texts_to_sequences([generated])[0]
        tokens = pad_sequences([tokens], maxlen=max_len, padding="pre")
        
        preds = model.predict(tokens, verbose=0)[0]
        idx = sample_token(preds, temperature=0.38, top_k=15, top_p=0.85, recent_tokens=history_indices)
        
        word = index_to_word.get(idx, "")
        if not word:
            break
            
        history_indices.append(idx)
        generated += " " + word
        
        if word.endswith((".", "!", "?")):
            break
            
    generated = generated[0].upper() + generated[1:]
    if not generated.endswith((".", "!", "?")):
        generated += "."
    return generated

# 3. Interactive Inputs & Prompts
CURATED_PROMPTS = [
    "The secret to success is",
    "In the middle of difficulty lies",
    "A person who never makes mistakes",
    "Courage is not the absence of fear",
    "The true measure of a man is",
    "Knowledge without wisdom is",
    "Happiness comes from within"
]

if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = CURATED_PROMPTS[0]

col_t1, col_t2 = st.columns([3, 1])
with col_t2:
    if st.button("🎲 Surprise Me", use_container_width=True):
        st.session_state.current_prompt = random.choice(CURATED_PROMPTS)
        st.rerun()

seed_text = col_t1.text_input(
    "Starting Phrase (Seed):",
    value=st.session_state.current_prompt,
    placeholder="e.g. The journey of...",
    label_visibility="collapsed"
)

# Horizontal Chips for Fast Selection
st.caption("Quick Starters:")
chip_cols = st.columns(3)
if chip_cols[0].button("💡 Secret to...", use_container_width=True):
    st.session_state.current_prompt = "The secret to"
    st.rerun()
if chip_cols[1].button("🌟 In the middle...", use_container_width=True):
    st.session_state.current_prompt = "In the middle of"
    st.rerun()
if chip_cols[2].button("🔥 Courage is...", use_container_width=True):
    st.session_state.current_prompt = "Courage is"
    st.rerun()

st.write("")

# Controls: Length & Num Variations
c1, c2 = st.columns(2)
with c1:
    quote_len = st.slider(
        "Max Length (words):",
        min_value=6,
        max_value=24,
        value=12,
        step=1,
        help="Shorter lengths (8-14) produce much more coherent proverbs."
    )
with c2:
    num_samples = st.radio(
        "Variations to Generate:",
        options=[1, 2, 3],
        index=0,
        horizontal=True
    )

# 4. Generation Execution
if st.button("Generate Quote ✨", type="primary", use_container_width=True):
    if not seed_text.strip():
        st.warning("Please provide a starting phrase.")
    else:
        with st.spinner("Generating quote candidates..."):
            st.markdown("### Results:")
            for i in range(num_samples):
                result = generate_quote(seed_text, quote_len)
                
                # Styled visual card
                st.markdown(f'<div class="quote-card">“{result}”</div>', unsafe_allow_html=True)
                
                # Copyable code block
                st.code(result, language=None)