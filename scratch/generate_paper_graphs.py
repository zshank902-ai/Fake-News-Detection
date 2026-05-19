import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

# Add parent directory to path to import model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import AdversarialFakeNewsModel

# Try importing seaborn, fallback to pure matplotlib style if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

class PretokenizedDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels, source_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.source_ids = source_ids
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'source_labels': torch.tensor(self.source_ids[idx], dtype=torch.long)
        }

def main():
    # Optimize CPU execution threads matching physical core capacity
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Dataset
    data_path = 'data/merged_multilingual_dataset.csv'
    if not os.path.exists(data_path):
        data_path = '../data/merged_multilingual_dataset.csv'
        if not os.path.exists(data_path):
            print(f"Dataset not found at data/merged_multilingual_dataset.csv")
            return
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows")
    
    # Sample exactly 15,000 rows (honoring "around 20k" on CPU)
    sample_size = min(15000, len(df))
    df_sample = df.sample(sample_size, random_state=42)
    print(f"Sampled {len(df_sample)} rows for evaluation")
    
    # Pre-tokenize all texts at once (extremely fast multi-threaded Rust tokenization)
    print("Pre-tokenizing all texts...")
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    encodings = tokenizer(
        df_sample['text'].astype(str).tolist(),
        padding=True,
        truncation=True,
        max_length=32, # Highly optimized sequence length for fast CPU run
        return_tensors='pt'
    )
    
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels = df_sample['label'].tolist()
    source_labels = df_sample['source_idx'].tolist() if 'source_idx' in df_sample.columns else [0] * len(df_sample)
    
    dataset = PretokenizedDataset(input_ids, attention_mask, labels, source_labels)
    loader = DataLoader(dataset, batch_size=32, pin_memory=True, num_workers=0)
    
    # 2. Load Model
    model = AdversarialFakeNewsModel(num_sources=101, lightweight=True).to(device)
    
    weights_path = 'final_model_lightweight.pt'
    if not os.path.exists(weights_path):
        weights_path = '../final_model_lightweight.pt'
        if not os.path.exists(weights_path):
            print(f"Weights file final_model_lightweight.pt not found.")
            return
            
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Loaded weights from {weights_path}")
    model.eval()
    
    # 3. Predict on sampled subset
    ground_truth = []
    probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating Predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)
            source_ids = batch['source_labels'].to(device)
            
            logits, _, _ = model(input_ids, attention_mask, source_ids=source_ids)
            probs = torch.sigmoid(logits).view(-1)
            
            ground_truth.extend(batch_labels.cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
            
    probabilities = np.array(probabilities)
    ground_truth = np.array(ground_truth)
    
    # Separate probabilities by true class
    probs_real = probabilities[ground_truth == 1]
    probs_fake = probabilities[ground_truth == 0]
    
    os.makedirs('reports', exist_ok=True)
    
    # --- GRAPH 1: Sorted Prediction Confidence Curve (Line Plot) ---
    print("Generating Sorted Prediction Confidence Curve...")
    plt.figure(figsize=(10, 6))
    
    sorted_real = np.sort(probs_real)
    sorted_fake = np.sort(probs_fake)
    
    x_real = np.linspace(0, 100, len(sorted_real))
    x_fake = np.linspace(0, 100, len(sorted_fake))
    
    plt.plot(x_real, sorted_real, label='True REAL News (Sorted Probs)', color='#2e7d32', linewidth=3)
    plt.plot(x_fake, sorted_fake, label='True FAKE News (Sorted Probs)', color='#c62828', linewidth=3)
    
    plt.title(f'Sorted Prediction Confidence Curve (N={sample_size:,})', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Sample Percentile (%)', fontsize=12)
    plt.ylabel('Model Prediction Probability (REAL)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    plt.savefig('reports/sorted_confidence_curve.png', dpi=300)
    plt.close()
    
    # --- GRAPH 2: Confidence Density Plot (KDE / Histogram) ---
    print("Generating Confidence Density Plot...")
    plt.figure(figsize=(10, 6))
    
    if HAS_SEABORN:
        sns.kdeplot(probs_real, label='True REAL News', color='#2e7d32', fill=True, alpha=0.3, linewidth=2.5)
        sns.kdeplot(probs_fake, label='True FAKE News', color='#c62828', fill=True, alpha=0.3, linewidth=2.5)
    else:
        plt.hist(probs_real, bins=50, alpha=0.5, label='True REAL News', color='#2e7d32', density=True)
        plt.hist(probs_fake, bins=50, alpha=0.5, label='True FAKE News', color='#c62828', density=True)
        
    plt.title(f'Prediction Probability Distribution (N={sample_size:,})', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Model Prediction Probability (REAL)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('reports/confidence_density_kde.png', dpi=300)
    plt.close()
    
    # --- GRAPH 3: ROC and Precision-Recall Curves ---
    print("Generating ROC and Precision-Recall Curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(ground_truth, probabilities)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='#1565c0', linewidth=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5)
    ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=11)
    ax1.set_ylabel('True Positive Rate (TPR)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=10, loc='lower right')
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(ground_truth, probabilities)
    ax2.plot(recall, precision, color='#8e24aa', linewidth=3, label='Precision-Recall Curve')
    ax2.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=11)
    ax2.set_ylabel('Precision (PPV)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=10, loc='lower left')
    
    plt.suptitle(f'ROC & Precision-Recall Performance (N={sample_size:,})', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('reports/performance_curves.png', dpi=300)
    plt.close()
    
    # --- GRAPH 4: Confusion Matrix ---
    print("Generating Confusion Matrix Heatmap...")
    cm = confusion_matrix(ground_truth, (probabilities > 0.5).astype(int))
    cm_norm = confusion_matrix(ground_truth, (probabilities > 0.5).astype(int), normalize='true')
    
    plt.figure(figsize=(8, 6))
    if HAS_SEABORN:
        labels = ['FAKE (Class 0)', 'REAL (Class 1)']
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels,
                    annot_kws={'size': 14, 'weight': 'bold'}, cbar=True)
    else:
        plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Normalized Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['FAKE', 'REAL'])
        plt.yticks(tick_marks, ['FAKE', 'REAL'])
        
        thresh = cm_norm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm_norm[i, j] > thresh else "black",
                         fontsize=14, fontweight='bold')
                
    plt.title(f'Confusion Matrix (N={sample_size:,})', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png', dpi=300)
    plt.close()
    
    print("\n" + "="*50)
    print("ALL GRAPHS GENERATED SUCCESSFULLY IN reports/ DIR:")
    print("1. reports/sorted_confidence_curve.png")
    print("2. reports/confidence_density_kde.png")
    print("3. reports/performance_curves.png")
    print("4. reports/confusion_matrix.png")
    print("="*50)

if __name__ == "__main__":
    main()
