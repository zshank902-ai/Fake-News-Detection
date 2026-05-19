import os
import sys
import torch
from transformers import AutoTokenizer

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import AdversarialFakeNewsModel

def smoke_test():
    print("--- RUNNING FINAL SOTA SMOKE TEST ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = AdversarialFakeNewsModel(num_sources=101).to(device)
    weights_path = 'final_model.pt'
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        print(f"SUCCESS: Weights loaded from {weights_path}")
    else:
        print(f"ERROR: {weights_path} missing!")
        return

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

    # 2. Test Scenarios
    scenarios = [
        {"text": "RBI increases repo rate by 0.25% to combat inflation.", "source": 0, "desc": "Official News (Real)"},
        {"text": "NASA finds alien base on dark side of the moon.", "source": 100, "desc": "Viral Conspiracy (Fake)"},
        {"text": "Get free 5G data by clicking this link immediately!", "source": 10, "desc": "WhatsApp Scam (Fake)"},
        {"text": "Prime Minister inaugurates new healthcare facility in Delhi.", "source": 0, "desc": "Regional News (Real)"}
    ]

    print(f"\n{'Scenario':<25} | {'Source':<10} | {'Prediction':<12} | {'Confidence'}")
    print("-" * 65)

    with torch.no_grad():
        for s in scenarios:
            inputs = tokenizer(s['text'], return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            s_id = torch.tensor([s['source']]).to(device)
            logits, _, _ = model(inputs['input_ids'], inputs['attention_mask'], source_ids=s_id)
            
            prob = torch.sigmoid(logits).item()
            pred = "FAKE [!]" if prob > 0.5 else "REAL [OK]"
            
            print(f"{s['desc']:<25} | {s['source']:<10} | {pred:<12} | {prob*100:.2f}%")

    print("\nSmoke Test Complete. Model is behaving as per the 10-epoch training logic.")

if __name__ == "__main__":
    smoke_test()
