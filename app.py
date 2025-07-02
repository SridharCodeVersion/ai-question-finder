import streamlit as st
import nltk
import os
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import pipeline

# Configure page
st.set_page_config(page_title="AI Question Generator", page_icon="📘")
st.title("📘 AI-Based Question Finder from Books")
st.markdown("Upload a textbook `.txt` file. The app will extract key points and generate questions automatically.")

@st.cache_resource
def load_question_generator():
    """Load the question generation model with error handling"""
    try:
        return pipeline("text2text-generation", model="valhalla/t5-base-e2e-qg")
    except Exception as e:
        st.error(f"Failed to load question generation model: {str(e)}")
        return None

def ensure_nltk_data():
    """Ensure NLTK punkt tokenizer is available"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt')
        except Exception as e:
            st.error(f"Failed to download NLTK data: {str(e)}")
            return False
    return True

# Initialize app
if not ensure_nltk_data():
    st.stop()

qg_model = load_question_generator()
if qg_model is None:
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload Text File", type=["txt"])

if uploaded_file is not None:
    try:
        text = uploaded_file.read().decode("utf-8")
        
        if not text.strip():
            st.warning("Uploaded file is empty.")
            st.stop()
            
        # Tokenize sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) < 1:
            st.warning("No sentences found in the text.")
            st.stop()
            
        # TF-IDF importance scoring
        vectorizer = TfidfVectorizer()
        try:
            X = vectorizer.fit_transform(sentences)
            scores = np.sum(X.toarray(), axis=1)
            top_indices = np.argsort(scores)[-5:]  # Get top 5 most important sentences
            important_sentences = [sentences[i] for i in top_indices]
        except Exception as e:
            st.error(f"Error processing text: {str(e)}")
            st.stop()
            
        # Generate questions
        st.info("Generating questions using AI...")
        st.subheader("🧠 Generated Questions:")
        
        with st.spinner("Processing..."):
            for sent in important_sentences:
                try:
                    input_text = "generate question: " + sent
                    ques = qg_model(input_text, max_length=50)[0]['generated_text']
                    st.markdown(f"• **{ques}**")
                except Exception as e:
                    st.warning(f"Couldn't generate question for: '{sent[:50]}...'")
                    
        st.success("Processing complete!")
        
    except UnicodeDecodeError:
        st.error("Failed to decode the file. Please upload a valid UTF-8 text file.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
