import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import pipeline

# Step 1: Load book content
with open("book.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Step 2: Sentence tokenization
sentences = sent_tokenize(text)

# Step 3: TF-IDF scoring
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)
scores = np.sum(X.toarray(), axis=1)
top_indices = np.argsort(scores)[-5:]
important_sentences = [sentences[i] for i in top_indices]

# Step 4: Question generation using T5
qg = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

print("\n📚 Top Generated Questions:\n")
for sent in important_sentences:
    input_text = "generate question: " + sent
    ques = qg(input_text)[0]['generated_text']
    print("•", ques)
