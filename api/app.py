# api/app.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from joblib import load
from src.preprocessing import clean_text, lemmatize_and_remove_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os

app = FastAPI(title="NLP Classification API")

# Load model and vectorizer
model = load("models/final_model.pkl")
vectorizer = load("models/vectorizer.pkl")  # Save this during training

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputText):
    cleaned = clean_text(data.text)
    lemmatized = lemmatize_and_remove_stopwords(cleaned)
    vector = vectorizer.transform([lemmatized])
    prediction = model.predict(vector)[0]
    return {"prediction": prediction}

@app.get("/")
def read_root():
    return {"message": "✅ NLP API is running"}
