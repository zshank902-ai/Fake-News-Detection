import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.model import AdversarialFakeNewsModel
from src.preprocess import FakeNewsDataset
import os
import time
from sklearn.metrics import f1_score

def train_sota_model(
    train_df, 
    val_df, 
    model_name='xlm-roberta-base', 
    batch_size=2, # Reduced from 8 to save RAM
    epochs=5, 
    lr=3e-5,
    grad_accum_steps=16 # Increased to maintain effective batch size of 32
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing SOTA Training on {device}")

    # --- Dataset & Loaders ---
    train_dataset = FakeNewsDataset(train_df, model_name=model_name)
    val_dataset = FakeNewsDataset(val_df, model_name=model_name)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=0)

    # --- Model Initialization ---
    model = AdversarialFakeNewsModel(model_name).to(device)
    
    # --- Optimizer & Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_loader) * epochs) // grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    # --- SOTA Components: AMP Scaler ---
    scaler = GradScaler()
    
    # --- Loss Functions ---
    criterion_label = nn.CrossEntropyLoss()
    criterion_lang = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    
    # Metrics for graphing
    epoch_losses = []
    epoch_accuracies = []
    epoch_f1_fake = []
    epoch_f1_real = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        loop = tqdm(train_loader, leave=True)
        for step, batch in enumerate(loop):
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            lang_labels = batch['lang_labels'].to(device)
            source_ids = batch['source_labels'].to(device)
            
            # --- Mixed Precision Forward ---
            with autocast():
                # Model returns (label_output, lang_logits, contrastive_features)
                label_output, lang_logits, _ = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    source_ids=source_ids,
                    alpha=1.0 # Adversarial alpha
                )
                
                # Check output shape. label_output might be [batch, 1] or [batch, 2]
                # AdversarialFakeNewsModel's classifier has num_classes=1 in __init__ by default
                # But CrossEntropy expects classes. If num_classes=1, we use BCE or update model.
                # Let's check model.py again. self.classifier last layer is Linear(256, num_classes)
                
                if label_output.size(1) == 1:
                    loss_label = nn.BCEWithLogitsLoss()(label_output.view(-1), labels.float())
                else:
                    loss_label = criterion_label(label_output, labels)
                
                # Language Discriminator loss
                loss_lang = criterion_lang(lang_logits, (lang_labels > 0).long()) # binary for simplicity: EN vs non-EN
                
                loss = (loss_label + 0.1 * loss_lang) / grad_accum_steps

            # --- Scaled Backward ---
            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0:
                # Gradient Clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * grad_accum_steps
            loop.set_description(f"Loss: {loss.item()*grad_accum_steps:.4f}")

        # --- Validation Phase ---
        val_acc, val_f1_fake, val_f1_real = evaluate_model(model, val_loader, device)
        avg_loss = total_loss/len(train_loader)
        print(f"Val Accuracy: {val_acc:.4f} | Val F1 (Fake): {val_f1_fake:.4f} | Val F1 (Real): {val_f1_real:.4f} | Avg Loss: {avg_loss:.4f}")
        
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(val_acc)
        epoch_f1_fake.append(val_f1_fake)
        epoch_f1_real.append(val_f1_real)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/sota_truth_shield.pt')
            print("New Best Model Saved!")

    # --- Generate Training Graph ---
    print("Generating training graph...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), epoch_losses, marker='o', label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), epoch_accuracies, marker='o', color='#1565c0', label='Val Accuracy')
    plt.plot(range(1, epochs+1), epoch_f1_fake, marker='s', color='#c62828', label='Val F1 (Fake)')
    plt.plot(range(1, epochs+1), epoch_f1_real, marker='^', color='#2e7d32', label='Val F1 (Real)')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs('reports', exist_ok=True)
    plt.savefig('reports/training_graph.png')
    print("Graph saved to reports/training_graph.png")

def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            source_ids = batch['source_labels'].to(device)
            
            label_output, _, _ = model(input_ids, attention_mask, source_ids=source_ids)
            
            if label_output.size(1) == 1:
                predicted = (torch.sigmoid(label_output).view(-1) > 0.5).long()
            else:
                _, predicted = torch.max(label_output, 1)
                
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
    accuracy = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels) if all_labels else 0.0
    f1_fake = f1_score(all_labels, all_preds, pos_label=0, zero_division=0)
    f1_real = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    
    return accuracy, f1_fake, f1_real

if __name__ == "__main__":
    print("⚠️ Please run run_all_training.py for full pipeline execution.")
