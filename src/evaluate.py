
# src/evaluate.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from wordcloud import WordCloud
import os
import joblib

def plot_confusion_matrix(y_true, y_pred, labels=None, save_path="outputs/figures/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax)
    plt.title("Confusion Matrix")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Confusion matrix saved to {save_path}")

def print_classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    print("Classification Report:\n", report)

def generate_wordcloud(text_series, save_path="outputs/figures/wordcloud.png"):
    text = " ".join(text_series)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud of Cleaned Text")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Word cloud saved to {save_path}")

if __name__ == "__main__":
    from src.evaluate import print_classification_metrics, plot_confusion_matrix
    from src.data_loader import load_raw_data

    model = joblib.load("models/final_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")  # ✅ Load encoder

    df = load_raw_data("clean_data.csv")
    X = vectorizer.transform(df["cleaned_text"])
    y_true = label_encoder.transform(df["label"])  # ✅ Encode labels
    y_pred = model.predict(X)

    print_classification_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)
