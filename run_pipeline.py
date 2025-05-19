
# run_pipeline.py

from src.data_loader import load_raw_data, save_cleaned_data
from src.preprocessing import preprocess_pipeline
from src.train import train_pipeline
from src.evaluate import print_classification_metrics, plot_confusion_matrix
import joblib

print("📥 STEP 1: Loading raw data...")
df = load_raw_data()

print("🧼 STEP 2: Preprocessing text...")
df = preprocess_pipeline(df, input_col="tweet", output_col="cleaned_text")
save_cleaned_data(df)

print("🏋️ STEP 3: Training the model...")
train_pipeline(model_name="SVM")  # Change to 'Voting', 'LogReg', etc. if desired

print("📈 STEP 4: Evaluating the model...")
model = joblib.load("models/final_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")  # ✅ Load label encoder

X = vectorizer.transform(df["cleaned_text"])
y = label_encoder.transform(df["label"])  # ✅ Encode labels to match model training
y_pred = model.predict(X)

print_classification_metrics(y, y_pred)
plot_confusion_matrix(y, y_pred)

print("✅ Pipeline completed successfully.")
