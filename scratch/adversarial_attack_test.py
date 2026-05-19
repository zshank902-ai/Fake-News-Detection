import torch
from transformers import AutoTokenizer
import os
import sys
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import AdversarialFakeNewsModel

def adversarial_test():
    print("--- ADVERSARIAL STRESS-TEST LAB ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = AdversarialFakeNewsModel(num_sources=101).to(device)
    weights_path = 'final_model.pt'
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        print(f"SOTA Brain Loaded for Testing.")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

    # Original Fake News
    original_text = "NASA discovers life on Mars after 100 years of searching. Exclusive photos released."
    
    # Adversarial Modifications (Synonym Swaps / Sneaky Words)
    # Goal: See if the model can be fooled by 'formal' or 'subtle' language
    adversarial_versions = [
        "NASA identifies biological signatures on Mars via planetary exploration. Visual data secured.",
        "A space agency has recently found evidence of extraterrestrial existence on the red planet. Images are leaked.",
        "Breaking: Revolutionary discovery of Martian life by scientists today. Verification in progress."
    ]

    print(f"\nOriginal Text: {original_text}")
    
    def get_prediction(text, s_id=10): # Using Source 10 (Suspicious)
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            logits, _, _ = model(inputs['input_ids'], inputs['attention_mask'], source_ids=torch.tensor([s_id]).to(device))
            prob = torch.sigmoid(logits).item()
        return "FAKE" if prob > 0.5 else "REAL", prob

    orig_pred, orig_prob = get_prediction(original_text)
    print(f"Original Prediction: {orig_pred} ({orig_prob*100:.2f}% Fake)")
    print("-" * 50)

    success_count = 0
    for i, adv_text in enumerate(adversarial_versions):
        pred, prob = get_prediction(adv_text)
        print(f"Adv Test {i+1}: {adv_text[:60]}...")
        print(f"Prediction: {pred} ({prob*100:.2f}% Fake)")
        
        if pred != orig_pred:
            print("ALERT: ATTACK SUCCESSFUL: Model was fooled!")
            success_count += 1
        else:
            print("OK: ATTACK FAILED: Model is robust.")
        print("-" * 30)

    vulnerability = (success_count / len(adversarial_versions)) * 100
    print(f"\nFINAL REPORT: Model Vulnerability Score: {vulnerability:.2f}%")
    if vulnerability < 20:
        print("RESULT: This model is ADVERSARIALLY ROBUST! 🛡️")
    else:
        print("RESULT: Model needs more Adversarial Hardening. 🥊")

if __name__ == "__main__":
    adversarial_test()
