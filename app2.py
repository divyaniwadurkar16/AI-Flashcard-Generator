import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# Download NLTK data
@st.cache_resource
def download_nltk():
    nltk.download("punkt")
    nltk.download("punkt_tab")


download_nltk()


# Load AI model
@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-base"
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base"
    )

    return tokenizer, model


tokenizer, model = load_model()


# Extract important sentences
def extract_important_sentences(text, num_sentences=5):

    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return sentences

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(sentences)

    scores = np.asarray(
        tfidf_matrix.sum(axis=1)
    ).ravel()

    important_indices = scores.argsort()[-num_sentences:]

    return [
        sentences[i]
        for i in important_indices
    ]


# Generate flashcards
def generate_flashcards(text, num_cards):

    flashcards = []

    sentences = extract_important_sentences(
        text,
        num_cards
    )

    for sentence in sentences:

        prompt = f"""
        Generate a clear educational question
        from this sentence:

        {sentence}
        """

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        )

        outputs = model.generate(
            **inputs,
            max_new_tokens=50
        )

        question = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        flashcards.append({
            "Question": question,
            "Answer": sentence
        })

    return flashcards



# ---------------- STREAMLIT UI ----------------

st.title("🤖 AI Flashcard Generator")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


num_cards = st.slider(
    "Number of Flashcards",
    1,
    20,
    5
)


if uploaded_file:

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text


    if st.button("Generate Flashcards"):

        with st.spinner("Generating flashcards..."):

            cards = generate_flashcards(
                text,
                num_cards
            )


        for i, card in enumerate(cards,1):

            st.subheader(
                f"Flashcard {i}"
            )

            st.write(
                "### Question"
            )

            st.write(
                card["Question"]
            )


            st.write(
                "### Answer"
            )

            st.write(
                card["Answer"]
            )
