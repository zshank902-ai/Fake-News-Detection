import torch
import sys
import os

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import AdversarialFakeNewsModel

def inspect_params():
    model = AdversarialFakeNewsModel(num_sources=101)
    
    print("=" * 60)
    print("PARAM INSPECTION")
    print("=" * 60)
    
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    # Group parameters by module
    modules_info = {}
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
            status = "Trainable"
        else:
            frozen_params += param_count
            status = "Frozen"
            
        # Get top-level module name
        parts = name.split('.')
        module_name = parts[0]
        if len(parts) > 1 and parts[0] == 'xlm_roberta':
            module_name = 'xlm_roberta.' + parts[1] # e.g. xlm_roberta.base_model
            if len(parts) > 2:
                module_name = 'xlm_roberta.base_model.' + parts[2]
        
        if module_name not in modules_info:
            modules_info[module_name] = {"trainable": 0, "frozen": 0}
        if param.requires_grad:
            modules_info[module_name]["trainable"] += param_count
        else:
            modules_info[module_name]["frozen"] += param_count
            
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"Frozen Parameters:    {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    
    print("\n--- Parameters Breakdown by Module ---")
    print(f"{'Module Name':<45} | {'Trainable Params':<18} | {'Frozen Params':<18}")
    print("-" * 87)
    for mod, info in sorted(modules_info.items()):
        print(f"{mod:<45} | {info['trainable']:<18,} | {info['frozen']:<18,}")

if __name__ == "__main__":
    inspect_params()
