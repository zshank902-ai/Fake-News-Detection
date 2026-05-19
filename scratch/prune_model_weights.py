import os
import sys
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import AdversarialFakeNewsModel

def prune_weights():
    weights_path = '../final_model.pt'
    output_path = '../final_model_lightweight.pt'
    
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found!")
        return
        
    print(f"Loading original weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location='cpu')
    print(f"Original keys count: {len(state_dict)}")
    
    # 1. Initialize a lightweight model to get target state dict keys and shapes
    print("Initializing lightweight model...")
    lightweight_model = AdversarialFakeNewsModel(lightweight=True)
    target_state_dict = lightweight_model.state_dict()
    
    new_state_dict = {}
    
    # 2. Map and slice keys
    for key, val in state_dict.items():
        # Skip unused modality layers
        if any(prefix in key for prefix in [
            'resnet.', 'vision_projector.', 'audio_projector.', 
            'video_projector.', 'knowledge_projector.',
            'empty_image', 'empty_audio', 'empty_video', 'empty_knowledge'
        ]):
            continue
            
        # Slice classifier.0.weight: [512, 1312] -> [512, 800]
        if key == 'classifier.0.weight':
            print(f"Slicing {key} from {val.shape} to {val[:, :800].shape}")
            new_state_dict[key] = val[:, :800]
            
        # Slice gating_network.0.weight: [5, 1312] -> [2, 800]
        elif key == 'gating_network.0.weight':
            print(f"Slicing {key} from {val.shape} to {val[:2, :800].shape}")
            new_state_dict[key] = val[:2, :800]
            
        # Slice gating_network.0.bias: [5] -> [2]
        elif key == 'gating_network.0.bias':
            print(f"Slicing {key} from {val.shape} to {val[:2].shape}")
            new_state_dict[key] = val[:2]
            
        else:
            # Keep other keys exactly as they are
            if key in target_state_dict:
                new_state_dict[key] = val
                
    # 3. Check for any missing keys
    missing_keys = set(target_state_dict.keys()) - set(new_state_dict.keys())
    extra_keys = set(new_state_dict.keys()) - set(target_state_dict.keys())
    
    if missing_keys:
        print(f"Warning: Missing keys in pruned state dict: {missing_keys}")
    if extra_keys:
        print(f"Warning: Extra keys in pruned state dict: {extra_keys}")
        
    # Verify shape matching
    for key in target_state_dict.keys():
        if key in new_state_dict:
            assert target_state_dict[key].shape == new_state_dict[key].shape, \
                f"Shape mismatch for {key}: target {target_state_dict[key].shape}, pruned {new_state_dict[key].shape}"
                
    print("Verification passed! Pruned state dict shapes match lightweight model exactly.")
    
    # 4. Save pruned state dict
    print(f"Saving lightweight weights to {output_path}...")
    torch.save(new_state_dict, output_path)
    
    orig_size = os.path.getsize(weights_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Original size: {orig_size:.2f} MB")
    print(f"Lightweight size: {new_size:.2f} MB (Reduction: {(1.0 - new_size/orig_size)*100:.2f}%)")

if __name__ == '__main__':
    prune_weights()
