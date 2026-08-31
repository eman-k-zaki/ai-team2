__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import re
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from google import genai

# ============ إعدادات الصفحة ============
st.set_page_config(page_title="NLP PDF Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 NLP PDF Chatbot")
st.caption("شات بوت يجاوب على أسئلتك بناءً على محتوى NLP باستخدام RAG + Gemini")

# ============ تحميل موارد NLTK (مرة واحدة فقط) ============
@st.cache_resource
def setup_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    return True

setup_nltk()

# ============ بناء قاعدة البيانات المتجهية (مرة واحدة فقط) ============
@st.cache_resource(show_spinner="جاري تجهيز قاعدة البيانات...")
def build_vector_db():
    documents = [
        Document(page_content="Natural Language Processing (NLP) is a field within artificial intelligence (AI) and computational linguistics focusing on interactions between computers and human language."),
        Document(page_content="Natural Language Understanding (NLU) focuses on comprehension, intent, and semantics. Natural Language Generation (NLG) creates coherent text from structured data."),
        Document(page_content="Text preprocessing steps include Lowercasing, Tokenization, Stopwords removal, Stemming, Lemmatization, POS tagging, and NER."),
        Document(page_content="Stemming removes word suffixes fast but may not return real words. Lemmatization uses lexical vocabulary and returns valid dictionary root words."),
        Document(page_content="Padding adds special tokens to equalize sequence length, while truncation cuts tokens exceeding maximum allowed length.")
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="nlp_preprocessing_rag"
    )
    return vector_db

vector_db = build_vector_db()

# ============ الحصول على مفتاح Gemini API ============
# الأولوية: Streamlit secrets (للنشر) ثم متغير بيئة (للتجربة المحلية)
api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))

if not api_key:
    st.error("⚠️ لم يتم العثور على مفتاح Gemini API. أضيفيه في Streamlit Secrets باسم GEMINI_API_KEY.")
    st.stop()

client = genai.Client(api_key=api_key)

# ============ دالة الرد على الأسئلة ============
def ask_pdf_bot(question, vector_database, k_chunks=3):
    results = vector_database.similarity_search(question, k=k_chunks)
    context_chunks = [doc.page_content for doc in results]
    context = "\n\n".join(context_chunks)

    prompt = f"""You are a helpful NLP expert assistant.
Answer the user question using ONLY the provided context below. If the answer is not contained within the context, politely state that you cannot find the answer in the document.

Context:
{context}

Question:
{question}

Answer:"""

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"حصل خطأ أثناء الاتصال بالـ LLM: {e}"

# ============ واجهة الشات ============
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض الرسائل السابقة
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# إدخال المستخدم
user_input = st.chat_input("اكتبي سؤالك هنا...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث والإجابة..."):
            answer = ask_pdf_bot(user_input, vector_db)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
