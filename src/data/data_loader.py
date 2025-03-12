import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from typing import Dict, List, Tuple
import numpy as np
import os

class ToxicCommentDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer: BertTokenizer, max_length: int = 128):
        # Convert texts to list if it's a pandas Series
        self.texts = texts.tolist() if isinstance(texts, pd.Series) else texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        
        # Handle unusual line terminators
        text = text.replace('\u2028', ' ').replace('\u2029', ' ')  # Remove line/paragraph separators
        text = ' '.join(text.splitlines())  # Normalize all newlines
        
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }

def load_toxic_data(data_path: str) -> Tuple[List[str], np.ndarray]:
    """Load and prepare the toxic comment dataset"""
    try:
        # Use encoding='utf-8-sig' to handle BOM if present
        df = pd.read_csv(data_path, encoding='utf-8-sig', on_bad_lines='skip')
        
        # List of toxicity categories
        toxic_categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        # Convert text column to list and labels to numpy array
        texts = df['comment_text'].tolist()
        labels = df[toxic_categories].values
        
        return texts, labels
    except Exception as e:
        raise RuntimeError(f"Error loading data from {data_path}: {str(e)}")

def create_data_loaders(
    texts: List[str],
    labels: np.ndarray,
    tokenizer: BertTokenizer,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    num_workers: int = 4  # Adjusted for Windows
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders"""
    try:
        # Calculate split index
        dataset_size = len(texts)
        train_size = int(dataset_size * train_ratio)
        
        # Split data
        train_texts = texts[:train_size]
        train_labels = labels[:train_size]
        val_texts = texts[train_size:]
        val_labels = labels[train_size:]
        
        # Create datasets
        train_dataset = ToxicCommentDataset(train_texts, train_labels, tokenizer)
        val_dataset = ToxicCommentDataset(val_texts, val_labels, tokenizer)
        
        # Create data loaders with Windows-optimized settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,  # Helps with CUDA performance
            persistent_workers=True  # Keeps workers alive between epochs
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        
        return train_loader, val_loader
    except Exception as e:
        raise RuntimeError(f"Error creating data loaders: {str(e)}")