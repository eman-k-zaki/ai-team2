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
st.set_page_config(page_title="Arabic to English Translator", page_icon="🌍")

st.title("🌍 Arabic to English Batch Translator")
st.write("Translate multiple Arabic sentences at once. Put each sentence on its own line.")

user_input = st.text_area(
    "Arabic Text (one sentence per line)",
    height=200,
    placeholder="اكتب كل جملة في سطر منفصل...\nمثال:\nمرحبا كيف حالك\nانا بحب البرمجة"
)

if st.button("Translate"):
    if user_input.strip() == "":
        st.warning("Please enter at least one sentence.")
    else:
        with st.spinner("Translating..."):
            output = translate(user_input)
        st.text_area("English Translations", value=output, height=200)