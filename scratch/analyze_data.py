import pandas as pd
import os

files = {
    "english_fake": "data/english_fake.csv",
    "english_real": "data/english_real.csv",
    "hindi": "data/hindi_dataset.csv",
    "urdu": "data/urdu_test.csv",
    "bengali": "data/bengali_test.csv"
}

for name, path in files.items():
    print(f"\n--- Analysis of {name} ({path}) ---")
    try:
        # Try different encodings for non-English data
        encoding = 'utf-8'
        if name in ['urdu', 'bengali', 'hindi']:
            encoding = 'utf-8-sig' # Handle BOM if present
        
        df = pd.read_csv(path, nrows=5, encoding=encoding)
        print(f"Columns: {df.columns.tolist()}")
        print("First row preview:")
        print(df.iloc[0].to_dict())
    except Exception as e:
        print(f"Error reading {path}: {e}")
        # Try latin-1 as fallback
        try:
            df = pd.read_csv(path, nrows=5, encoding='latin-1')
            print(f"Fallback Latin-1 Columns: {df.columns.tolist()}")
        except:
            print("Failed with fallback encoding too.")
