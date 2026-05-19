import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import re
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, XLMRobertaModel
import torchvision.models as models
from peft import LoraConfig, get_peft_model
from urllib.parse import urlparse

# --- 1. Preprocessing ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\s+|www\.\s+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class MultilingualDataset(Dataset):
    def __init__(self, encodings, labels, lang_labels, source_labels):
        self.encodings = encodings
        self.labels = labels
        self.lang_labels = lang_labels
        self.source_labels = source_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['lang_labels'] = torch.tensor(self.lang_labels[idx])
        item['source_labels'] = torch.tensor(self.source_labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# --- 2. Model Architecture ---
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)

class SuperSotaModel(nn.Module):
    def __init__(self, model_name='xlm-roberta-base', num_sources=101):
        super(SuperSotaModel, self).__init__()
        base_model = XLMRobertaModel.from_pretrained(model_name)
        
        # LoRA Config
        peft_config = LoraConfig(r=8, lora_alpha=32, target_modules=["query", "value"], lora_dropout=0.05, bias="none")
        self.xlm_roberta = get_peft_model(base_model, peft_config)
        
        self.source_embedding = nn.Embedding(num_sources, 32)
        
        # Self-Attention
        self.self_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(768 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        self.lang_discriminator = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask, source_ids=None, alpha=1.0):
        outputs = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        attn_output, _ = self.self_attention(sequence_output, sequence_output, sequence_output)
        pooled_output = attn_output[:, 0, :]
        
        source_feats = self.source_embedding(source_ids) if source_ids is not None else torch.zeros((pooled_output.size(0), 32)).to(pooled_output.device)
        
        combined = torch.cat((pooled_output, source_feats), dim=1)
        label_out = self.classifier(combined)
        
        reverse_feats = grad_reverse(pooled_output, alpha)
        lang_out = self.lang_discriminator(reverse_feats)
        
        return label_out, lang_out

# --- 3. Main Training Script ---
def run_colab_training():
    print("Colab Training Initialized...")
    # Installation (Run in Colab cell): !pip install transformers peft lime
    
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    
    # LOAD DATA (Assuming CSVs are in 'data/' folder in Colab)
    en_fake = pd.read_csv('data/english_fake.csv')
    en_real = pd.read_csv('data/english_real.csv')
    hi_df = pd.read_csv('data/hindi_dataset.csv', encoding='latin-1')
    
    # Extraction & Standardizing (Simplified for Colab)
    def extract_source(url):
        try: return urlparse(url).netloc.replace('www.', '')
        except: return 'unknown'
        
    en_fake['source'] = en_fake['news_url'].apply(extract_source)
    en_real['source'] = en_real['news_url'].apply(extract_source)
    en_fake['label'], en_real['label'] = 1, 0
    en_fake['language'], en_real['language'] = 0, 0
    
    hi_df['label'] = hi_df['Label'].apply(lambda x: 1 if str(x).upper() in ['FALSE', 'FAKE'] else 0)
    hi_df['language'] = 1
    hi_df['source'] = hi_df['Web'].fillna('unknown')
    
    en_df = pd.concat([en_fake, en_real])[['title', 'label', 'language', 'source']]
    hi_df = hi_df[['Statement', 'label', 'language', 'source']].rename(columns={'Statement': 'title'})
    df = pd.concat([en_df, hi_df]).reset_index(drop=True)
    df['title'] = df['title'].apply(clean_text)
    
    top_sources = df['source'].value_counts().index[:100]
    df['source_idx'] = df['source'].apply(lambda x: list(top_sources).index(x) if x in top_sources else 100)
    
    # Tokenization
    encodings = tokenizer(df['title'].tolist(), padding='max_length', truncation=True, max_length=128)
    
    dataset = MultilingualDataset(encodings, df['label'].values, df['language'].values, df['source_idx'].values)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True) # Colab has better RAM, can use batch_size 16
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SuperSotaModel(num_sources=len(top_sources)+1).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion_label = nn.BCEWithLogitsLoss()
    criterion_lang = nn.CrossEntropyLoss()
    
    scaler = torch.amp.GradScaler()
    
    print("Starting Super Intelligent Training on Cloud...")
    for epoch in range(10): # 10 Epochs for SOTA
        model.train()
        for i, batch in enumerate(train_loader):
            ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            labels, lang_labels = batch['labels'].to(device), batch['lang_labels'].to(device)
            source_labels = batch['source_labels'].to(device)
            
            with torch.amp.autocast(device_type='cuda'):
                label_out, lang_out = model(ids, mask, source_ids=source_labels)
                loss = criterion_label(label_out.view(-1), labels.float()) + 0.1 * criterion_lang(lang_out, lang_labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1} | Step {i} | Loss: {loss.item():.4f}")
        
        torch.save(model.state_dict(), f"super_sota_epoch_{epoch+1}.pt")
    print("Training Complete! Weights saved.")

if __name__ == "__main__":
    run_colab_training()
