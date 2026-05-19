import torch
from transformers import AutoTokenizer
import sys
import os

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import AdversarialFakeNewsModel

def diagnose_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = AdversarialFakeNewsModel(num_sources=101).to(device)
    weights_path = '../final_model.pt'
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Weights missing at {weights_path}")
        return

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    
    scenarios = [
        {"text": "RBI increases repo rate by 0.25% to combat inflation.", "source": 0, "desc": "Official News (Real)"},
        {"text": "NASA finds alien base on dark side of the moon.", "source": 100, "desc": "Viral Conspiracy (Fake)"}
    ]
    
    # Let's inspect some classifier weights
    print("\n--- Classifier Weights Inspection ---")
    last_linear = model.classifier[-1]
    print(f"Classifier last layer: {last_linear}")
    print(f"Weight (min/max/mean): {last_linear.weight.min().item():.4f} / {last_linear.weight.max().item():.4f} / {last_linear.weight.mean().item():.4f}")
    print(f"Bias: {last_linear.bias.item():.4f}")
    
    first_linear = model.classifier[0]
    print(f"Classifier first layer weight (min/max/mean): {first_linear.weight.min().item():.4f} / {first_linear.weight.max().item():.4f} / {first_linear.weight.mean().item():.4f}")
    print(f"Classifier first layer bias (min/max/mean): {first_linear.bias.min().item():.4f} / {first_linear.bias.max().item():.4f} / {first_linear.bias.mean().item():.4f}")

    with torch.no_grad():
        for s in scenarios:
            print("\n" + "=" * 50)
            print(f"Scenario: {s['desc']} - '{s['text']}'")
            print("=" * 50)
            
            inputs = tokenizer(s['text'], return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            s_id = torch.tensor([s['source']]).to(device)
            
            # Forward pass step by step
            # 1. Text Features
            outputs = model.xlm_roberta(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            sequence_output = outputs.last_hidden_state
            
            # Multihead attention
            attn_output, _ = model.self_attention(sequence_output, sequence_output, sequence_output)
            pooled_output = attn_output[:, 0, :]
            print(f"pooled_output (min/max/mean): {pooled_output.min().item():.4f} / {pooled_output.max().item():.4f} / {pooled_output.mean().item():.4f}")
            
            # Source embedding
            source_feats = model.source_embedding(s_id)
            print(f"source_feats (min/max/mean): {source_feats.min().item():.4f} / {source_feats.max().item():.4f} / {source_feats.mean().item():.4f}")
            
            # Gating network decision
            image_features = model.empty_image.expand(pooled_output.size(0), -1)
            audio_features = model.empty_audio.expand(pooled_output.size(0), -1)
            video_features = model.empty_video.expand(pooled_output.size(0), -1)
            knowledge_features = model.empty_knowledge.expand(pooled_output.size(0), -1)
            
            raw_combined = torch.cat((pooled_output, source_feats, image_features, audio_features, video_features, knowledge_features), dim=1)
            gate_weights = model.gating_network(raw_combined)
            print(f"gate_weights: {gate_weights[0].tolist()}")
            
            # Combined features
            combined_features = raw_combined * gate_weights.repeat_interleave(torch.tensor([768, 32, 128, 128, 256]).to(device), dim=1)
            print(f"combined_features (min/max/mean): {combined_features.min().item():.4f} / {combined_features.max().item():.4f} / {combined_features.mean().item():.4f}")
            
            # Classifier forward
            x = model.classifier[0](combined_features)
            print(f"After classifier[0] (min/max/mean): {x.min().item():.4f} / {x.max().item():.4f} / {x.mean().item():.4f}")
            x = model.classifier[1](x) # ReLU
            x = model.classifier[2](x) # Dropout
            x = model.classifier[3](x)
            print(f"After classifier[3] (min/max/mean): {x.min().item():.4f} / {x.max().item():.4f} / {x.mean().item():.4f}")
            x = model.classifier[4](x) # ReLU
            x = model.classifier[5](x) # Dropout
            logits = model.classifier[6](x)
            print(f"Final logit: {logits.item():.4f}")
            prob = torch.sigmoid(logits).item()
            print(f"Final probability: {prob*100:.2f}%")

if __name__ == "__main__":
    diagnose_model()
