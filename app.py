import streamlit as st
import nltk
import os
import nltk.data

# ✅ Ensure punkt is downloaded (for Streamlit Cloud)
nltk_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_path):
    os.makedirs(nltk_path)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_path)

# Set NLTK path (important for Streamlit Cloud)
nltk.data.path.append(nltk_path)

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import pipeline

st.set_page_config(page_title="AI Question Generator", page_icon="📘")

st.title("📘 AI-Based Question Finder from Books")
st.markdown("Upload a textbook `.txt` file. The app will extract key points and generate questions automatically.")

uploaded_file = st.file_uploader("Upload Text File", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    sentences = sent_tokenize(text)

    # TF-IDF importance scoring
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    scores = np.sum(X.toarray(), axis=1)
    top_indices = np.argsort(scores)[-5:]
    important_sentences = [sentences[i] for i in top_indices]

    # Load question generation model
    st.info("Generating questions using AI...")
    qg = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

    st.subheader("🧠 Generated Questions:")
    for sent in important_sentences:
        input_text = "generate question: " + sent
        ques = qg(input_text)[0]['generated_text']
        st.markdown(f"• **{ques}**")

    st.success("Done! You can upload another file to generate more questions.")
