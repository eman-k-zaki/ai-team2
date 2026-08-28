# -*- coding: utf-8 -*-
# ============================================
# Arabic to English Batch Translator - Streamlit App
# ============================================

import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# ============================================
# STEP 1: Load the model and tokenizer (cached so it loads ONCE only)
# ============================================
@st.cache_resource
def load_model():
    model_name = "Helsinki-NLP/opus-mt-ar-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()


# ============================================
# STEP 2: Build the BATCH translation function
# ============================================
def translate(text_block):
    # Split input into separate sentences (one per line)
    sentences = [line.strip() for line in text_block.split("\n") if line.strip() != ""]

    if len(sentences) == 0:
        return "Please enter at least one sentence."

    # Tokenize all sentences together as a batch (padding=True is needed here)
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

    # Generate translations for the whole batch at once
    translated_tokens = model.generate(**inputs)

    # Decode each translated sentence
    translations = [
        tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens
    ]

    # Pair each original sentence with its translation
    result = "\n".join(
        f"{i+1}. {sentences[i]}  ->  {translations[i]}"
        for i in range(len(sentences))
    )

    return result


# ============================================
# STEP 3: Streamlit UI
# ============================================
st.set_page_config(page_title="Arabic to English Translator", page_icon="🌍", layout="centered")
 
st.title("🌍 Arabic to English Translator")
st.caption("Type a sentence and press Enter to translate it instantly.")
 
st.divider()
 
# Keep a running history of translations across Enter presses
if "history" not in st.session_state:
    st.session_state.history = []  # list of (arabic, english) tuples
 
# text_input inside a form submits automatically when the user presses Enter
with st.form(key="translate_form", clear_on_submit=True):
    user_input = st.text_input(
        "Arabic sentence",
        placeholder="اكتب جملة واحدة وادوس Enter...",
        help="Press Enter to translate."
    )
    submitted = st.form_submit_button("🔁 Translate", use_container_width=True)
 
if submitted:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a sentence.")
    else:
        with st.spinner("Translating..."):
            english = translate(user_input).split("  ->  ", 1)[-1]
        st.session_state.history.insert(0, (user_input, english))
 
# Show translation history (most recent first)
if st.session_state.history:
    st.divider()
    col_a, col_b = st.columns([5, 1])
    with col_a:
        st.subheader("Translations")
    with col_b:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.history = []
            st.rerun()
 
    for arabic, english in st.session_state.history:
        st.markdown(f"**AR:** {arabic}")
        st.markdown(f"**EN:** {english}")
        st.divider()
 