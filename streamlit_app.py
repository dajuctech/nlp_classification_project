# Streamlit dashboard
import streamlit as st
import joblib
from src.preprocessing import clean_text

st.title("Tweet Sentiment Classifier")

model = joblib.load('models/final_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

text_input = st.text_area("Enter a tweet:")
if st.button("Classify"):
    clean = clean_text(text_input)
    vector = vectorizer.transform([clean])
    prediction = model.predict(vector)
    st.write(f"Predicted Label: {prediction[0]}")
