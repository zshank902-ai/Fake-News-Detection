import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import re
import string
from urllib.parse import urlparse

class FakeNewsDataset(Dataset):
    def __init__(self, df, model_name='xlm-roberta-base', max_len=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.texts = df['text'].astype(str).tolist()
        self.labels = df['label'].tolist()
        self.lang_labels = df['lang'].tolist() if 'lang' in df.columns else [0] * len(df)
        self.source_labels = df['source_idx'].tolist() if 'source_idx' in df.columns else [0] * len(df)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        lang = self.lang_labels[idx]
        source = self.source_labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'lang_labels': torch.tensor(lang, dtype=torch.long),
            'source_labels': torch.tensor(source, dtype=torch.long)
        }

class MultilingualFakeNewsDataset(Dataset):
    # Kept for compatibility if used elsewhere
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

def clean_text(text):
    """Advanced text cleaning for multilingual data."""
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Basic cleaning
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_dataloader(df, model_name='xlm-roberta-base', batch_size=16, shuffle=True):
    dataset = FakeNewsDataset(df, model_name=model_name)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
