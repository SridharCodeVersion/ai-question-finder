# AI Question Generator

An application that automatically generates questions from text documents using natural language processing.

## Features

- Upload any .txt document
- Automatic extraction of key sentences
- AI-generated questions based on content
- Downloadable results
- Responsive interface

## How to Use

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

3. Upload a text file and view the generated questions

## Deployment

To deploy on Streamlit Cloud:

1. Create a new repository with these files
2. Connect your Streamlit account to the repository
3. Set the main file path to `app.py`
4. Deploy!

## Technical Details

- Uses NLTK for text processing
- TF-IDF for sentence importance scoring
- T5 transformer model for question generation
