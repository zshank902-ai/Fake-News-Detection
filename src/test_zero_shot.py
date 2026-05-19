import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report
from model import BaselineFakeNewsModel, AdversarialFakeNewsModel
from preprocess import setup_tokenizer, clean_data, tokenize_data
import pandas as pd

def zero_shot_evaluation(model_baseline, model_adversarial, test_loader, language_name="Unseen", device='cpu'):
    print(f"\n--- Zero-Shot Evaluation on {language_name} ---")
    
    # Evaluate Baseline
    model_baseline.eval()
    baseline_preds = []
    labels_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            ids, mask, labels = [b.to(device) for b in batch]
            out = model_baseline(ids, mask)
            preds = (out.squeeze() > 0.5).int()
            baseline_preds.extend(preds.cpu().numpy() if preds.ndim > 0 else [preds.item()])
            labels_list.extend(labels.cpu().numpy())
            
    f1_baseline = f1_score(labels_list, baseline_preds)
    
    # Evaluate Adversarial
    model_adversarial.eval()
    adv_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            ids, mask, labels = [b.to(device) for b in batch]
            out, _ = model_adversarial(ids, mask)
            preds = (out.squeeze() > 0.5).int()
            adv_preds.extend(preds.cpu().numpy() if preds.ndim > 0 else [preds.item()])
            
    f1_adv = f1_score(labels_list, adv_preds)
    
    print(f"Baseline F1-Score:    {f1_baseline:.4f}")
    print(f"Adversarial F1-Score: {f1_adv:.4f}")
    
    improvement = ((f1_adv - f1_baseline) / f1_baseline) * 100 if f1_baseline > 0 else 0
    print(f"Improvement: {improvement:.2f}%")
    
    if improvement > 0:
        print(f"WIN! The adversarial model generalizes better to {language_name} (Zero-Shot).")
    else:
        print(f"Adversarial model did not improve on {language_name}. Check hyperparameters.")

    return f1_baseline, f1_adv

if __name__ == "__main__":
    # Placeholder for loading Urdu (UMC) or Bengali (BanFakeNews)
    # df_urdu = pd.read_csv('data/umc_urdu_test.csv')
    # ... process and tokenize ...
    print("Zero-Shot script ready. Waiting for Urdu/Bengali test data.")
