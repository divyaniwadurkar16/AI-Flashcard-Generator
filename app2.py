
import nltk
from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

def main():
    # Download NLTK resources if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')

    try:
        nltk.data.find('tokenizers/punkt_tab')
    except nltk.downloader.DownloadError:
        nltk.download('punkt_tab')

    # Initialize the text generation pipeline
    generator = pipeline(
        "text-generation",
        model="google/flan-t5-base"
    )

    # Assume a PDF file named 'sample-local-pdf.pdf' is present in the same directory
    filename = 'sample-local-pdf.pdf'

    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found. Please ensure the PDF file is in the same directory as app.py.")
        return

    # Read text from PDF
    reader = PdfReader(filename)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    if not text.strip():
        print(f"Error: No text extracted from '{filename}'. Please check the PDF content.")
        return

    # Define helper function to extract important sentences
    def extract_important_sentences(text, num_sentences=5):
        sentences = sent_tokenize(text)
        if not sentences:
            return []

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        scores = np.asarray(tfidf_matrix.sum(axis=1)).ravel()
        important_indices = scores.argsort()[-num_sentences:]
        important_sentences = [
            sentences[i]
            for i in important_indices
        ]
        return important_sentences

    # Define main flashcard generation function
    def generate_flashcards(text, num_cards=5):
        flashcards = []
        important_sentences = extract_important_sentences(
            text,
            num_cards
        )

        for sentence in important_sentences:
            prompt = f"""
            Generate a clear educational question
            from this sentence:

            {sentence}
            """
            result = generator(
                prompt,
                max_length=50
            )
            question = result[0]["generated_text"]

            flashcards.append({
                "Question": question,
                "Answer": sentence
            })
        return flashcards

    # Get number of flashcards from user
    try:
        num = int(input("How many flashcards do you want? "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    cards = generate_flashcards(text, num)

    # Print flashcards
    for i, card in enumerate(cards, 1):
        print(f"\nFlashcard {i}")
        print("Question:")
        print(card["Question"])
        print()
        print("Answer:")
        print(card["Answer"])
        print("-"*50)

if __name__ == "__main__":
    main()
