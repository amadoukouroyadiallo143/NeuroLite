"""
Module de chargement optimisé des données pour NeuroLite
"""
from datasets import load_dataset
from torch.utils.data import Dataset
import os

class NeuroLiteDataset(Dataset):
    """Classe personnalisée pour le chargement du dataset"""
    def __init__(self, dataset_name="tyzhu/wikitext-103-raw-v1-sent-permute-9", cache_dir=None):
        self.dataset = load_dataset(
            dataset_name, 
            cache_dir=cache_dir or os.path.expanduser("~/.cache/neurolite")
        )
        
    def __len__(self):
        return len(self.dataset["train"])
    
    def __getitem__(self, idx):
        return self.dataset["train"][idx]


def prepare_dataset(dataset, tokenizer, max_length=512):
    """Tokenize et prépare le dataset"""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True
        )
    
    return dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=["text"]
    )
