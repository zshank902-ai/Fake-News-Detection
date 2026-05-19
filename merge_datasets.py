import pandas as pd
import os
from urllib.parse import urlparse

def load_csv(path):
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding='latin1')

def extract_source(url):
    try:
        return urlparse(str(url)).netloc.replace('www.', '')
    except:
        return 'unknown'

def merge_datasets():
    data_dir = "d:/Python Workshop/fake_news_detection/data"
    merged_path = os.path.join(data_dir, "merged_multilingual_dataset.csv")
    
    all_data = []

    # 1. English Fake
    print("Processing English Fake...")
    en_fake_path = os.path.join(data_dir, "english_fake.csv")
    if os.path.exists(en_fake_path):
        df = load_csv(en_fake_path)
        df['text'] = df['title']
        df['label'] = 0
        df['lang'] = 0 # English
        df['source'] = df['news_url'].apply(extract_source)
        all_data.append(df[['text', 'label', 'lang', 'source']])

    # 2. English Real
    print("Processing English Real...")
    en_real_path = os.path.join(data_dir, "english_real.csv")
    if os.path.exists(en_real_path):
        df = load_csv(en_real_path)
        df['text'] = df['title']
        df['label'] = 1
        df['lang'] = 0 # English
        df['source'] = df['news_url'].apply(extract_source)
        all_data.append(df[['text', 'label', 'lang', 'source']])

    # 3. Hindi
    print("Processing Hindi...")
    hindi_path = os.path.join(data_dir, "hindi_dataset.csv")
    if os.path.exists(hindi_path):
        df = load_csv(hindi_path)
        df['text'] = df['Statement']
        df['label'] = df['Label'].apply(lambda x: 1 if str(x).strip().upper() == 'TRUE' else 0)
        df['lang'] = 1 # Hindi
        df['source'] = df['Web'].fillna('unknown')
        all_data.append(df[['text', 'label', 'lang', 'source']])

    # 4. Bengali
    print("Processing Bengali...")
    bengali_path = os.path.join(data_dir, "bengali_test.csv")
    if os.path.exists(bengali_path):
        df = load_csv(bengali_path)
        df['text'] = df['Headline'].fillna('') + " " + df['Content'].fillna('')
        df['label'] = df['Label'].apply(lambda x: 1 if str(x).strip() == '1' else 0)
        df['lang'] = 2 # Bengali
        df['source'] = 'unknown'
        all_data.append(df[['text', 'label', 'lang', 'source']])

    # 5. Urdu
    print("Processing Urdu...")
    urdu_path = os.path.join(data_dir, "urdu_test.csv")
    if os.path.exists(urdu_path):
        df = load_csv(urdu_path)
        df['text'] = df['News Items']
        df['label'] = df['Label'].apply(lambda x: 1 if str(x).strip().upper() == 'TRUE' else 0)
        df['lang'] = 3 # Urdu
        df['source'] = 'unknown'
        all_data.append(df[['text', 'label', 'lang', 'source']])

    if not all_data:
        print("No data found to merge!")
        return

    print("Merging all data...")
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # Clean up
    merged_df.dropna(subset=['text'], inplace=True)
    merged_df = merged_df[merged_df['text'].astype(str).str.strip() != ""]
    
    # Encode Source (Top 100, rest 100)
    print("Encoding sources...")
    top_sources = merged_df['source'].value_counts().index[:100]
    source_to_idx = {s: i for i, s in enumerate(top_sources)}
    merged_df['source_idx'] = merged_df['source'].apply(lambda x: source_to_idx.get(x, 100))
    
    print(f"Total samples: {len(merged_df)}")
    print(f"Label distribution:\n{merged_df['label'].value_counts()}")
    
    merged_df.to_csv(merged_path, index=False, encoding='utf-8-sig')
    print(f"Saved merged dataset to {merged_path}")

if __name__ == "__main__":
    merge_datasets()
