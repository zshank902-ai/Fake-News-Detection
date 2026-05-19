import os
import sys

# --- PROFESSOR'S CRITICAL PATH FIX ---
# This ensures that 'src' is always findable, regardless of where you run the script from.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model import AdversarialFakeNewsModel
from src.preprocess import MultilingualFakeNewsDataset, clean_text
from transformers import AutoTokenizer
import pandas as pd
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_loader(df, tokenizer, batch_size=8):
    df['title'] = df['title'].fillna('').apply(clean_text)
    encodings = tokenizer(df['title'].tolist(), truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    labels = df['label'].tolist()
    lang_labels = df['language'].tolist()
    source_labels = df['source_idx'].tolist()
    
    dataset = MultilingualFakeNewsDataset(encodings, labels, lang_labels, source_labels)
    return DataLoader(dataset, batch_size=batch_size)

def run_ablation_variant(name, train_df, val_df, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[RUNNING] Ablation Variant: {name}")
    
    # Initialize Model with 2 classes for CrossEntropy
    model = AdversarialFakeNewsModel(num_classes=2).to(device)
    
    if config.get('no_grl'):
        alpha = 0.0
    else:
        alpha = 1.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    train_loader = prepare_loader(train_df, tokenizer, batch_size=8)
    val_loader = prepare_loader(val_df, tokenizer, batch_size=8)
    
    start_time = time.time()
    
    for epoch in range(1): # Reduced to 1 epoch for quick graph generation
        model.train()
        for batch in tqdm(train_loader, desc=f"{name}"):
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['labels'].to(device)
            
            label_out, lang_logits, _ = model(input_ids, attention_mask, alpha=alpha)
            loss = criterion(label_out, labels)
            
            if not config.get('no_grl'):
                lang_labels = batch['lang_labels'].to(device)
                lang_loss = criterion(lang_logits, lang_labels)
                loss += 0.1 * lang_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['labels'].to(device)
            
            label_out, _, _ = model(input_ids, attention_mask)
            _, predicted = torch.max(label_out, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / total
    inference_time = (time.time() - start_time) / (len(val_loader) * 8)
    
    return {
        "Variant": name,
        "Accuracy": f"{accuracy*100:.2f}%",
        "Inf. Latency (avg)": f"{inference_time*1000:.2f}ms",
        "Accuracy_Val": accuracy * 100,
        "Latency_Val": inference_time * 1000
    }

if __name__ == "__main__":
    os.makedirs('reports', exist_ok=True)
    
    try:
        # Load Hindi with robust encoding
        hindi_path = os.path.join(project_root, 'data', 'hindi_dataset.csv')
        try:
            hindi_df = pd.read_csv(hindi_path, encoding='utf-8').sample(50)
        except:
            hindi_df = pd.read_csv(hindi_path, encoding='latin-1').sample(50)
            
        hindi_df['label'] = hindi_df['Label'].apply(lambda x: 1 if str(x).upper() == 'FALSE' else 0)
        hindi_df['language'] = 1
        hindi_df['source_idx'] = 0
        hindi_df = hindi_df.rename(columns={'Statement': 'title'})

        # Load English
        english_path = os.path.join(project_root, 'data', 'english_fake.csv')
        english_df = pd.read_csv(english_path).sample(30)
        english_df['label'] = 1
        english_df['language'] = 0
        english_df['source_idx'] = 0
        
        train_df = pd.concat([hindi_df[:35], english_df[:20]])
        val_df = pd.concat([hindi_df[35:], english_df[20:]])
        
        results = []
        results.append(run_ablation_variant("Full Shield", train_df, val_df, {}))
        results.append(run_ablation_variant("No GRL", train_df, val_df, {"no_grl": True}))
        results.append(run_ablation_variant("Text-Only", train_df, val_df, {"text_only": True}))
        
        df_results = pd.DataFrame(results)

        # Ensure reports directory exists
        reports_dir = os.path.join(project_root, 'reports')
        if not os.path.exists(reports_dir): os.makedirs(reports_dir)

        # Accuracy Graph
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        ax = sns.barplot(x="Variant", y="Accuracy_Val", data=df_results, palette="viridis")
        plt.title("Ablation Study: Model Accuracy Comparison")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        acc_path = os.path.join(reports_dir, 'ablation_accuracy.png')
        plt.savefig(acc_path)
        print(f"\n[SUCCESS] Accuracy Graph Saved: {acc_path}")

        # Latency Graph
        plt.figure(figsize=(10, 6))
        ax2 = sns.barplot(x="Variant", y="Latency_Val", data=df_results, palette="magma")
        plt.title("Inference Latency: Speed Comparison")
        plt.ylabel("Latency (ms)")
        lat_path = os.path.join(reports_dir, 'ablation_latency.png')
        plt.savefig(lat_path)
        print(f"[SUCCESS] Latency Graph Saved: {lat_path}")

        print("\n" + "="*50)
        print("FINAL RESULTS TABLE")
        print("="*50)
        print(df_results[['Variant', 'Accuracy', 'Inf. Latency (avg)']].to_markdown(index=False))
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
