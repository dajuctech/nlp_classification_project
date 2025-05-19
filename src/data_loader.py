# src/data_loader.py

import pandas as pd
import os

def load_raw_data(filename="train.csv", folder="data/raw"):
    path = os.path.join(folder, filename)
    df = pd.read_csv(path)
    print(f"✅ Loaded data from {path} with shape: {df.shape}")
    return df

def save_cleaned_data(df, filename="clean_data.csv", folder="data/processed"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    df.to_csv(path, index=False)
    print(f"✅ Cleaned data saved to {path}")

if __name__ == "__main__":
    from src.data_loader import load_raw_data, save_cleaned_data
    df = load_raw_data()
    save_cleaned_data(df)
