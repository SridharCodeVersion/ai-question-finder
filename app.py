import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import pipeline
import time
from typing import List, Tuple

# Configure the app
st.set_page_config(
    page_title="AI Question Generator",
    page_icon="❓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("📚 AI Question Generator from Documents")
st.markdown("""
Upload any text document (.txt file) and the AI will automatically:
1. Extract key sentences
2. Generate relevant questions
3. Present them in an organized way
""")

# Initialize NLTK
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

setup_nltk()

@st.cache_resource(show_spinner=False)
def load_question_generator():
    """Load and cache the question generation model"""
    try:
        return pipeline(
            "text2text-generation",
            model="valhalla/t5-small-e2e-qg",
            device="cpu"
        )
    except Exception as e:
        st.error(f"Failed to load question generation model: {str(e)}")
        return None

# Initialize the model
with st.spinner("Loading AI model (this may take a minute)..."):
    qg_model = load_question_generator()

if qg_model is None:
    st.stop()

def process_text(text: str) -> Tuple[List[str], List[str]]:
    """
    Process the input text to extract important sentences and generate questions
    
    Args:
        text: Input text content
        
    Returns:
        Tuple of (important_sentences, generated_questions)
    """
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    if len(sentences) < 3:
        return sentences, []
    
    # Calculate importance scores using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        X = vectorizer.fit_transform(sentences)
        scores = np.sum(X.toarray(), axis=1)
        top_indices = np.argsort(scores)[-5:]  # Get top 5 important sentences
        important_sentences = [sentences[i].strip() for i in top_indices]
    except:
        important_sentences = sentences[:5]  # Fallback if TF-IDF fails
    
    # Generate questions
    generated_questions = []
    for sent in important_sentences:
        try:
            input_text = f"generate question: {sent}"
            question = qg_model(
                input_text,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )[0]['generated_text']
            generated_questions.append(question.strip())
        except:
            continue
    
    return important_sentences, generated_questions

# Main app interface
def main():
    uploaded_file = st.file_uploader(
        "Upload a text document (.txt)",
        type=["txt"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        try:
            text = uploaded_file.read().decode("utf-8")
            
            if not text.strip():
                st.warning("The uploaded file is empty.")
                return
            
            with st.spinner("Analyzing document and generating questions..."):
                start_time = time.time()
                important_sentences, questions = process_text(text)
                processing_time = time.time() - start_time
            
            st.success(f"Processed document in {processing_time:.1f} seconds")
            
            if not questions:
                st.warning("Couldn't generate questions from this document.")
                return
            
            # Display results in tabs
            tab1, tab2 = st.tabs(["Generated Questions", "Key Sentences"])
            
            with tab1:
                st.subheader("🧠 AI-Generated Questions")
                for i, question in enumerate(questions, 1):
                    st.markdown(f"{i}. **{question}**")
                
                # Download button for questions
                questions_text = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
                st.download_button(
                    "Download Questions",
                    questions_text,
                    file_name="generated_questions.txt",
                    mime="text/plain"
                )
            
            with tab2:
                st.subheader("🔑 Key Sentences Identified")
                for i, sentence in enumerate(important_sentences, 1):
                    st.markdown(f"{i}. {sentence}")
            
        except UnicodeDecodeError:
            st.error("Failed to read the file. Please upload a valid UTF-8 text file.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
