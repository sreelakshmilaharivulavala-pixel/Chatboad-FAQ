import json
import re
import string
from collections import Counter

import numpy as np
import spacy
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data (run once)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__)

# Load NLP models
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))

# Load FAQs
with open('faq_data.json', 'r') as f:
    faq_data = json.load(f)

faq_questions = [faq['question'] for faq in faq_data['faqs']]
faq_answers = [faq['answer'] for faq in faq_data['faqs']]


def preprocess_text(text):
    """Preprocess using NLP libraries (NLTK + SpaCy)."""
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', ' ', text)
    # NLTK tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    # SpaCy lemmatization
    doc = nlp(' '.join(tokens))
    lemmas = [token.lemma_ for token in doc if not token.is_stop and len(token.lemma_) > 2]
    return ' '.join(lemmas)

# Precompute TF-IDF matrix for FAQs
preprocessed_faqs = [preprocess_text(q) for q in faq_questions]
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
faq_tfidf = vectorizer.fit_transform(preprocessed_faqs)

@app.route('/')
def index():
    return render_template('index.html', product=faq_data['product'])

@app.route('/api/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '').strip()
    if not user_input:
        return jsonify({"answer": "Please ask a question about the AeroBook Pro.", "confidence": 0.0})

    # Preprocess user input
    processed_input = preprocess_text(user_input)
    input_tfidf = vectorizer.transform([processed_input])

    # Cosine similarity
    similarities = cosine_similarity(input_tfidf, faq_tfidf).flatten()
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    # Intent threshold: if very low similarity, return fallback
    if best_score < 0.15:
        return jsonify({
            "answer": "I'm not sure about that. Please ask something specific about battery, warranty, storage, ports, display, weight, returns, or compatibility for the AeroBook Pro.",
            "confidence": best_score,
            "matched_faq": faq_questions[best_idx]
        })

    return jsonify({
        "answer": faq_answers[best_idx],
        "confidence": round(best_score, 3),
        "matched_faq": faq_questions[best_idx]
    })

@app.route('/api/faqs')
def get_faqs():
    return jsonify({"faqs": faq_data['faqs'], "count": len(faq_data['faqs'])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
