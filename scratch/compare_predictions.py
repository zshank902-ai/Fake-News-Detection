import torch
import pandas as pd
import sys
import os
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import AdversarialFakeNewsModel

def compare_predictions():
    # Configure stdout to handle utf-8 on Windows console
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    data_path = '../data/merged_multilingual_dataset.csv'
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    # Split exactly like in training
    _, val_df = train_test_split(df, test_size=0.1, random_state=42)
    val_sample = val_df.sample(500, random_state=42)
    
    # Load Model
    model = AdversarialFakeNewsModel(num_sources=101, lightweight=True).to(device)
    weights_path = '../final_model_lightweight.pt'
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Loaded lightweight model weights.")
    else:
        print(f"Model weights not found at {weights_path}.")
        return
        
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    
    # In the dataset, 1 is REAL and 0 is FAKE
    # 5 Real News samples
    reals = val_sample[val_sample['label'] == 1].head(5)
    # 5 Fake News samples
    fakes = val_sample[val_sample['label'] == 0].head(5)
    
    print("\n" + "=" * 100)
    print("ANALYZING TRUE REAL SAMPLES (Expected Probability > 50% for REAL)")
    print("=" * 100)
    for idx, row in reals.iterrows():
        text = row['text']
        src_idx = int(row['source_idx'])
        lang = int(row['lang'])
        
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            # Actual source index prediction
            logits_act, _, _ = model(inputs['input_ids'], inputs['attention_mask'], source_ids=torch.tensor([src_idx]).to(device))
            prob_act = torch.sigmoid(logits_act).item()
            
            # Neutral source index (2) prediction
            logits_neu, _, _ = model(inputs['input_ids'], inputs['attention_mask'], source_ids=torch.tensor([2]).to(device))
            prob_neu = torch.sigmoid(logits_neu).item()
            
            # Source 1 (mostly Real) prediction
            logits_real_src, _, _ = model(inputs['input_ids'], inputs['attention_mask'], source_ids=torch.tensor([1]).to(device))
            prob_real_src = torch.sigmoid(logits_real_src).item()
            
        print(f"Text:  '{text}'")
        print(f"Lang:  {lang} | Actual Source Idx: {src_idx}")
        print(f"Prob (Actual Source):  {prob_act*100:.2f}% (Verdict: {'REAL' if prob_act > 0.5 else 'FAKE'})")
        print(f"Prob (Neutral Source): {prob_neu*100:.2f}% (Verdict: {'REAL' if prob_neu > 0.5 else 'FAKE'})")
        print(f"Prob (Real Source 1):  {prob_real_src*100:.2f}% (Verdict: {'REAL' if prob_real_src > 0.5 else 'FAKE'})")
        print("-" * 100)

    print("\n" + "=" * 100)
    print("ANALYZING TRUE FAKE SAMPLES (Expected Probability < 50% for REAL)")
    print("=" * 100)
    for idx, row in fakes.iterrows():
        text = row['text']
        src_idx = int(row['source_idx'])
        lang = int(row['lang'])
        
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            # Actual source index prediction
            logits_act, _, _ = model(inputs['input_ids'], inputs['attention_mask'], source_ids=torch.tensor([src_idx]).to(device))
            prob_act = torch.sigmoid(logits_act).item()
            
            # Neutral source index (2) prediction
            logits_neu, _, _ = model(inputs['input_ids'], inputs['attention_mask'], source_ids=torch.tensor([2]).to(device))
            prob_neu = torch.sigmoid(logits_neu).item()
            
            # Source 1 (mostly Real) prediction
            logits_real_src, _, _ = model(inputs['input_ids'], inputs['attention_mask'], source_ids=torch.tensor([1]).to(device))
            prob_real_src = torch.sigmoid(logits_real_src).item()
            
        print(f"Text:  '{text}'")
        print(f"Lang:  {lang} | Actual Source Idx: {src_idx}")
        print(f"Prob (Actual Source):  {prob_act*100:.2f}% (Verdict: {'REAL' if prob_act > 0.5 else 'FAKE'})")
        print(f"Prob (Neutral Source): {prob_neu*100:.2f}% (Verdict: {'REAL' if prob_neu > 0.5 else 'FAKE'})")
        print(f"Prob (Real Source 1):  {prob_real_src*100:.2f}% (Verdict: {'REAL' if prob_real_src > 0.5 else 'FAKE'})")
        print("-" * 100)

if __name__ == "__main__":
    compare_predictions()
