import os
import pandas as pd

def analyze_datasets():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    print(f"Data Directory: {data_dir}\n")
    
    files = {
        "English Fake News": "english_fake.csv",
        "English Real News": "english_real.csv",
        "Hindi Dataset": "hindi_dataset.csv",
        "Urdu Test Dataset": "urdu_test.csv",
        "Bengali Test Dataset": "bengali_test.csv",
        "Merged Multilingual Dataset": "merged_multilingual_dataset.csv"
    }
    
    for label_name, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"[MISSING] {label_name} ({filename}) does not exist.")
            continue
            
        print("=" * 80)
        print(f"DATASET: {label_name} ({filename})")
        print("=" * 80)
        print(f"File size: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
        
        # Load the dataset
        try:
            # For non-English datasets, try utf-8-sig or latin-1
            encoding = 'utf-8'
            if filename in ['hindi_dataset.csv', 'urdu_test.csv', 'bengali_test.csv']:
                encoding = 'utf-8-sig'
            
            try:
                df = pd.read_csv(filepath, encoding=encoding)
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding='latin-1')
                
            print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"Columns: {df.columns.tolist()}")
            
            print("\nNull Values:")
            nulls = df.isnull().sum()
            for col, count in nulls.items():
                if count > 0:
                    print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
            if nulls.sum() == 0:
                print("  None")
                
            # Class distribution
            label_col = None
            for col in ['label', 'Label', 'Verdict', 'verdict', 'class']:
                if col in df.columns:
                    label_col = col
                    break
            
            if label_col:
                print(f"\nLabel Distribution ({label_col}):")
                counts = df[label_col].value_counts(dropna=False)
                for val, count in counts.items():
                    print(f"  {val}: {count} ({count/len(df)*100:.2f}%)")
            else:
                print("\nNo label column found.")
                
            # Language distribution
            lang_col = None
            for col in ['language', 'lang', 'Language']:
                if col in df.columns:
                    lang_col = col
                    break
            if lang_col:
                print(f"\nLanguage Distribution ({lang_col}):")
                counts = df[lang_col].value_counts(dropna=False)
                for val, count in counts.items():
                    print(f"  {val}: {count} ({count/len(df)*100:.2f}%)")
            
            print("\nSample Rows:")
            print(df.head(2).to_string(index=False))
            print()
            
        except Exception as e:
            print(f"Error analyzing {filename}: {e}\n")

if __name__ == "__main__":
    analyze_datasets()
