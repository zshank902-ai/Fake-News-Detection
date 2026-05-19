import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import AdversarialFakeNewsModel
import torch

def debug_loading():
    device = torch.device("cpu")
    model = AdversarialFakeNewsModel(num_sources=101)
    weights_path = 'final_model.pt'
    
    checkpoint = torch.load(weights_path, map_location=device)
    
    # Load and capture missing/unexpected keys
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    
    print("--- LOAD DEBUG REPORT ---")
    print(f"Total Checkpoint Keys: {len(checkpoint)}")
    print(f"Missing Keys (in model but not in checkpoint): {len(missing_keys)}")
    print(f"Unexpected Keys (in checkpoint but not in model): {len(unexpected_keys)}")
    
    if len(missing_keys) > 0:
        print("\nSample Missing Keys:")
        for k in missing_keys[:10]:
            print(f"  - {k}")
            
    if len(unexpected_keys) > 0:
        print("\nSample Unexpected Keys:")
        for k in unexpected_keys[:10]:
            print(f"  - {k}")

    # Check LoRA activation
    lora_keys = [k for k in checkpoint.keys() if 'lora' in k]
    print(f"\nLoRA Keys in Checkpoint: {len(lora_keys)}")
    
    model_lora_keys = [k for k in model.state_dict().keys() if 'lora' in k]
    print(f"LoRA Keys in Model: {len(model_lora_keys)}")

if __name__ == "__main__":
    debug_loading()
