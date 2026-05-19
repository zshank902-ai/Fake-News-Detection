import torch
import json

weights = torch.load('final_model.pt', map_location='cpu')

info = {}
if isinstance(weights, dict):
    info["main_keys"] = list(weights.keys())
    if 'model_state_dict' in weights:
        info["found_state_dict"] = True
        info["sample_keys"] = list(weights['model_state_dict'].keys())[:50]
    else:
        info["found_state_dict"] = False
        info["sample_keys"] = list(weights.keys())[:50]
else:
    info["type"] = str(type(weights))

with open('scratch/model_info.json', 'w') as f:
    json.dump(info, f, indent=4)
print("Saved info to scratch/model_info.json")
