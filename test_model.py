import torch
from transformers import AutoTokenizer
from src.model import AdversarialFakeNewsModel
import sys
import os

# Terminal Formatting
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def predict_news(text, model_path='final_model_lightweight.pt', model_name='xlm-roberta-base'):
    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Model
    if not os.path.exists(model_path):
        print(f"{Colors.FAIL}Error: Model file '{model_path}' not found. Please ensure weight conversion has run.{Colors.ENDC}")
        return

    model = AdversarialFakeNewsModel(model_name, lightweight=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 4. Preprocess Input
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    source_ids = torch.tensor([0]).to(device) # Default source

    # 5. Inference
    with torch.no_grad():
        label_output, _, _ = model(input_ids, attention_mask, source_ids=source_ids)
        
        # Binary Classification Logic (0 = FAKE, 1 = REAL)
        if label_output.size(1) == 1:
            prob = torch.sigmoid(label_output).item()
            prediction = "REAL" if prob > 0.5 else "FAKE"
            confidence = prob if prediction == "REAL" else 1.0 - prob
        else:
            probs = torch.softmax(label_output, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            prediction = "REAL" if pred_idx.item() == 1 else "FAKE"
            confidence = confidence.item()

    # 6. Beautiful Output
    print("\n" + "="*50)
    print(f"{Colors.BOLD}TRUTH SHIELD AI - ANALYSIS RESULT{Colors.ENDC}")
    print("="*50)
    print(f"{Colors.OKBLUE}Input Text:{Colors.ENDC} {text[:100]}...")
    
    color = Colors.FAIL if prediction == "FAKE" else Colors.OKGREEN
    print(f"{Colors.BOLD}Verdict:   {color}{prediction}{Colors.ENDC}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    print(f"{Colors.HEADER}--- TRUTH SHIELD INFERENCE SYSTEM ---{Colors.ENDC}")
    while True:
        user_input = input(f"\n{Colors.BOLD}Paste News Text (or type 'exit' to quit): {Colors.ENDC}")
        if user_input.lower() == 'exit':
            break
        if len(user_input.strip()) < 5:
            print(f"{Colors.WARNING}Please enter a valid news snippet.{Colors.ENDC}")
            continue
        
        predict_news(user_input)
