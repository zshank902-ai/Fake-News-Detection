import os
import sys
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import AdversarialFakeNewsModel
from src.preprocess import FakeNewsDataset

def evaluate_on_val():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Dataset
    data_path = '../data/merged_multilingual_dataset.csv'
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows")
    
    # Split exactly like in training
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    # Take a sample of validation data to speed up evaluation (e.g., 500 samples)
    val_sample = val_df.sample(500, random_state=42)
    print(f"Validation sample size: {len(val_sample)}")
    
    # Create DataLoader
    val_dataset = FakeNewsDataset(val_sample, model_name='xlm-roberta-base')
    val_loader = DataLoader(val_dataset, batch_size=8, pin_memory=True, num_workers=0)
    
    # 2. Load Model
    model = AdversarialFakeNewsModel(num_sources=101, lightweight=True).to(device)
    weights_path = '../final_model_lightweight.pt'
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"SUCCESS: Loaded weights from {weights_path}")
    else:
        print(f"ERROR: {weights_path} missing!")
        return
        
    model.eval()
    
    # Evaluate
    correct = 0
    total = 0
    predictions = []
    ground_truth = []
    probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            source_ids = batch['source_labels'].to(device)
            
            logits, _, _ = model(input_ids, attention_mask, source_ids=source_ids)
            
            probs = torch.sigmoid(logits).view(-1)
            predicted = (probs > 0.5).long()
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            predictions.extend(predicted.cpu().tolist())
            ground_truth.extend(labels.cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
            
    accuracy = correct / total
    print("\n" + "="*50)
    print(f"EVALUATION RESULTS ON VAL SAMPLE (N={total})")
    print("="*50)
    print(f"Accuracy: {accuracy*100:.2f}%")
    
    # Compute majority class accuracy (predicting 1 for all, since REAL is the majority class)
    majority_class = 1
    majority_acc = sum(1 for label in ground_truth if label == majority_class) / total
    print(f"Majority Class (REAL) Accuracy: {majority_acc*100:.2f}%")
    
    # Summary of predictions (0 = FAKE, 1 = REAL)
    fake_preds = sum(1 for p in predictions if p == 0)
    real_preds = sum(1 for p in predictions if p == 1)
    print(f"Predicted FAKE (0): {fake_preds} ({fake_preds/total*100:.2f}%)")
    print(f"Predicted REAL (1): {real_preds} ({real_preds/total*100:.2f}%)")
    
    # Average probabilities (probability is for class 1, which is REAL)
    avg_prob_real_class = sum(p for p, l in zip(probabilities, ground_truth) if l == 1) / sum(ground_truth)
    avg_prob_fake_class = sum(p for p, l in zip(probabilities, ground_truth) if l == 0) / (total - sum(ground_truth))
    print(f"Average predicted probability of REAL for True REAL: {avg_prob_real_class*100:.2f}%")
    print(f"Average predicted probability of REAL for True FAKE: {avg_prob_fake_class*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    evaluate_on_val()
