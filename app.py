import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

# ✅ Ensure punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ✅ Streamlit UI config
st.set_page_config(page_title="AI Question Generator", page_icon="📘")

st.title("📘 AI-Based Question Finder from Books")
st.markdown("Upload a textbook `.txt` file. The app will extract key points and generate questions automatically.")

# ✅ Upload file
uploaded_file = st.file_uploader("Upload Text File", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    sentences = sent_tokenize(text)

    # ✅ TF-IDF Scoring
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    scores = np.sum(X.toarray(), axis=1)
    top_indices = np.argsort(scores)[-5:]
    important_sentences = [sentences[i] for i in top_indices]

    # ✅ Load AI question generation model
    st.info("Generating questions using AI...")
    qg = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

    st.subheader("🧠 Generated Questions:")
    for sent in important_sentences:
        input_text = "generate question: " + sent
        ques = qg(input_text)[0]['generated_text']
        st.markdown(f"• **{ques}**")

    st.success("Done! Upload another file to generate more.")
