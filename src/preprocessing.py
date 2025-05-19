# src/preprocessing.py

import re
import html
import emoji
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Cleans text by removing special characters, emojis, URLs, and repeated characters."""
    text = html.unescape(text)
    text = contractions.fix(text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'(https?://|www\.)\S+|[a-zA-Z0-9.-]+\.(com|org|net|edu|gov|uk|io|co)\S*', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s:]', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(cleaned)

def preprocess_pipeline(df, input_col="tweet", output_col="cleaned_text"):
    df[output_col] = df[input_col].astype(str).apply(clean_text).apply(lemmatize_and_remove_stopwords)
    return df
