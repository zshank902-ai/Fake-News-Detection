import torch

def inspect_weights(filepath):
    print(f"Loading weights from {filepath}...")
    try:
        state_dict = torch.load(filepath, map_location='cpu')
        print(f"Loaded successfully! Total keys: {len(state_dict)}")
        
        # Print first 20 keys
        print("\n--- FIRST 20 KEYS ---")
        keys = list(state_dict.keys())
        for k in keys[:20]:
            print(f"  {k}: shape {state_dict[k].shape if hasattr(state_dict[k], 'shape') else 'no shape'}")
            
        # Print some specific classifier keys
        print("\n--- CLASSIFIER KEYS ---")
        classifier_keys = [k for k in keys if 'classifier' in k]
        for k in classifier_keys:
            print(f"  {k}: shape {state_dict[k].shape if hasattr(state_dict[k], 'shape') else 'no shape'}")
            
        # Print some specific gating keys
        gating_keys = [k for k in keys if 'gating' in k]
        print(f"\n--- GATING KEYS ({len(gating_keys)}) ---")
        for k in gating_keys:
            print(f"  {k}")
            
        # Check for vision/resnet keys
        resnet_keys = [k for k in keys if 'resnet' in k or 'vision' in k]
        print(f"\n--- RESNET/VISION KEYS ({len(resnet_keys)}) ---")
        for k in resnet_keys[:10]:
            print(f"  {k}")
            
    except Exception as e:
        print(f"Error loading {filepath}: {e}")

if __name__ == "__main__":
    inspect_weights("../final_model.pt")
