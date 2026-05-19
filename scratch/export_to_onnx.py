import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.onnx
from src.model import AdversarialFakeNewsModel

def export_onnx():
    print("--- SOTA ONNX EXPORT ENGINE ---")
    device = torch.device("cpu") # Exporting for CPU/Universal inference
    model = AdversarialFakeNewsModel(num_sources=101).to(device)
    model.eval()

    # Load weights if exist
    if os.path.exists('final_model.pt'):
        model.load_state_dict(torch.load('final_model.pt', map_location=device), strict=False)

    # Prepare dummy inputs (XLM-R expects input_ids and attention_mask)
    dummy_input_ids = torch.zeros((1, 128), dtype=torch.long).to(device)
    dummy_attention_mask = torch.ones((1, 128), dtype=torch.long).to(device)
    dummy_source_ids = torch.tensor([10]).to(device)
    
    # Placeholders for multimodal features (matching our Super-Omniscient arch)
    dummy_img = torch.zeros((1, 3, 224, 224)).to(device)
    dummy_audio = torch.zeros((1, 1024)).to(device)
    dummy_video = torch.zeros((1, 2048)).to(device)
    dummy_knowledge = torch.zeros((1, 768)).to(device)

    export_path = "models/sota_truth_shield.onnx"
    if not os.path.exists('models'): os.makedirs('models')

    print(f"Exporting model to {export_path}...")
    
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask, dummy_img, dummy_audio, dummy_video, dummy_knowledge, dummy_source_ids),
        export_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask', 'images', 'audio_feats', 'video_feats', 'knowledge_feats', 'source_ids'],
        output_names=['logits', 'lang_logits', 'contrastive_feats'],
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'source_ids': {0: 'batch_size'}
        }
    )
    
    print(f"Success! ONNX Model is ready for High-Speed Inference.")

if __name__ == "__main__":
    export_onnx()
