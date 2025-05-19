# src/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from joblib import dump

from src.model import get_models
from src.preprocessing import preprocess_pipeline

def train_pipeline(input_path="data/processed/clean_data.csv", model_name="SVM"):
    # Load data
    df = pd.read_csv(input_path)
    
    # Optional: preprocess if not already done
    if 'cleaned_text' not in df.columns:
        df = preprocess_pipeline(df, input_col="tweet")

    # ✅ Drop rows with missing cleaned_text
    df = df.dropna(subset=["cleaned_text"])

    # Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["cleaned_text"])
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])

    # Balance classes
    X_resampled, y_resampled = SMOTE(random_state=100).fit_resample(X, y)

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=100)

    # Get model
    model = get_models()[model_name]
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    print(f"\nAccuracy: {accuracy_score(y_val, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_val, y_pred))

    # Save model and vectorizer
    dump(model, "models/final_model.pkl")
    dump(vectorizer, "models/vectorizer.pkl")  # ✅ Save vectorizer for Streamlit
    print("✅ Model saved to models/final_model.pkl")
    print("✅ Vectorizer saved to models/vectorizer.pkl")

if __name__ == "__main__":
    train_pipeline(model_name="SVM")
