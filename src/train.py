# Training logic
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.data_loader import load_data
from src.preprocessing import clean_text
from src.model import build_model
import pandas as pd

def train_pipeline(data_path, text_col='text', label_col='label'):
    df = load_data(data_path)
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(df[text_col], df[label_col], test_size=0.2, random_state=42)

    model = build_model()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    return model
