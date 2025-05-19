# streamlit_app.py

import streamlit as st
import joblib
from src.preprocessing import clean_text

# Page config
st.set_page_config(page_title="Tweet Sentiment Classifier", layout="centered")
st.title("🤖 Tweet Sentiment Classifier")

# Load model and vectorizer
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/final_model.pkl")
        vectorizer = joblib.load("models/vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Error loading model/vectorizer: {e}")
        return None, None

model, vectorizer = load_model()

# Text input
text_input = st.text_area("📝 Enter a tweet for classification:")

# Classify button
if st.button("🔍 Classify"):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text first.")
    elif model and vectorizer:
        cleaned = clean_text(text_input)
        X = vectorizer.transform([cleaned])
        pred = model.predict(X)
        st.success(f"✅ Predicted Label: **{pred[0]}**")
    else:
        st.error("⚠️ Model or vectorizer not loaded properly.")
