"""
Script de pré-entraînement NeuroLite - Version Professionnelle
"""

import os
import time
from datetime import datetime
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import argparse
from data_loader import NeuroLiteDataset, prepare_dataset
from torch.utils.tensorboard import SummaryWriter

from neurolite import (
    NeuroLiteModel,
    NeuroLiteConfig
)

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Configuration du logger
class TrainingLogger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "training.log")
        
    def log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
        print(message)

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# Configuration Distributed
def setup_distributed():
    dist.init_process_group('nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

# Hyperparameter Tuning
class HyperparameterTuner:
    @staticmethod
    def get_lr_schedule(lr, warmup_steps, total_steps):
        """Schedule triangulaire pour le learning rate"""
        return lambda step: lr * min(1, step/warmup_steps) * min(1, (total_steps - step)/total_steps)

# Fonction d'entraînement optimisée
def train_model(args, model, tokenizer, train_dataloader, eval_dataloader, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Configuration de l'optimiseur
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps
    )
    
    # Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    
    # Mémoire GPU
    logger.log(f"Mémoire GPU allouée: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(progress_bar):
            if args.test_mode and step >= args.max_test_steps:
                break
                
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
            epoch_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.1e}",
                'gpu': f"{torch.cuda.memory_allocated()/1024**2:.0f}MB"
            })
        
        # Validation
        eval_loss = evaluate(model, eval_dataloader, device, args)
        logger.log(f"Epoch {epoch+1} - Train Loss: {epoch_loss/len(train_dataloader):.4f} | Eval Loss: {eval_loss:.4f}")
        
        # Sauvegarde
        if eval_loss < best_loss:
            best_loss = eval_loss
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            logger.log(f"Checkpoint saved to {args.output_dir}")

# Fonction d'évaluation
def evaluate(model, dataloader, device, args):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
    
    return total_loss / len(dataloader)

# Fonction de chargement des données
def load_datasets(data_dir, test_mode):
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", data_dir=data_dir)
    if test_mode:
        dataset = dataset.shuffle(seed=42).select(range(1000))
    
    train_dataset, eval_dataset = dataset["train"].train_test_split(test_size=0.01, seed=42)
    
    return train_dataset, eval_dataset

# Fonction de préparation des données
def prepare_dataset(args, tokenizer, dataset):
    column_names = dataset.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        # tokenizer.encode_batch returns a list of Encoding objects
        encodings = tokenizer.encode_batch(
            examples[text_column_name],
            add_special_tokens=True  # Important for model to learn special tokens if defined
        )
        # datasets.map expects a dictionary of lists
        return {"input_ids": [encoding.ids for encoding in encodings]}

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing texts",
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can customize this part.
        if total_length >= args.max_seq_length:
            total_length = (total_length // args.max_seq_length) * args.max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + args.max_seq_length] for i in range(0, total_length, args.max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy() # For causal LM, labels are shifted in the model or loss function
        return result

    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts into chunks of {args.max_seq_length}",
    )
    return lm_dataset

# Fonction de chargement du modèle
def load_model(args):
    config = NeuroLiteConfig.from_json_file(args.config_file)
    tokenizer = NeuroLiteTokenizer.from_pretrained(config.tokenizer_name)
    model = NeuroLiteModel(config)
    
    return model, tokenizer

# Fonction de chargement du tokenizer
def load_tokenizer(args):
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        print("Training a new BPE tokenizer...")
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        
        trainer = BpeTrainer(vocab_size=args.vocab_size, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])
        
        def text_iterator(dataset_split):
            for i in range(0, len(dataset_split), 1000):
                yield dataset_split[i: i + 1000]["text"]

        tokenizer.train_from_iterator(text_iterator(dataset["train"]), trainer=trainer)
        os.makedirs(args.output_dir, exist_ok=True)
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")
    
    return tokenizer

# Nouvelle fonction d'entraînement
def train(args):
    # Initialisation distributed si nécessaire
    if args.distributed:
        local_rank = setup_distributed()
    else:
        local_rank = 0
    
    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir) if local_rank == 0 else None
    
    # Early Stopping
    early_stopper = EarlyStopping(patience=args.patience) if args.early_stop else None
    
    # Chargement des données
    dataset = NeuroLiteDataset()
    train_dataset = prepare_dataset(dataset, tokenizer)
    
    if args.distributed:
        sampler = DistributedSampler(train_dataset)
    else:
        sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers
    )
    
    # Entraînement distribué
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)
    
    # Hyperparameter tuning
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        HyperparameterTuner.get_lr_schedule(args.lr, args.warmup_steps, len(train_loader)*args.epochs)
    )
    
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            outputs = model(**batch)
            loss = outputs.loss
            
            if writer:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                
            # Early stopping
            if early_stopper and early_stopper(val_loss):
                print("Early stopping triggered!")
                break

if __name__ == "__main__":
    # Configuration des arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="models/neurolite-pretrained")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--max_test_steps", type=int, default=100)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--distributed", action="store_true", help="Activer l'entraînement distribué")
    parser.add_argument("--early_stop", action="store_true", help="Activer l'early stopping")
    parser.add_argument("--patience", type=int, default=3, help="Patience pour l'early stopping")
    parser.add_argument("--log_dir", type=str, default="logs", help="Répertoire pour TensorBoard")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Nombre de pas de warmup")
    parser.add_argument("--num_workers", type=int, default=4, help="Nombre de workers pour le chargement des données")
    
    args = parser.parse_args()
    
    # Initialisation
    logger = TrainingLogger(args.output_dir)
    logger.log(f"Starting NeuroLite pretraining with args: {args}")
    
    # Chargement des données
    train_dataset, eval_dataset = load_datasets(args.data_dir, args.test_mode)
    
    # Chargement du tokenizer
    tokenizer = load_tokenizer(args)
    
    # Préparation des données
    train_dataset_tok = prepare_dataset(args, tokenizer, train_dataset)
    eval_dataset_tok = prepare_dataset(args, tokenizer, eval_dataset)
    
    # Chargement du modèle
    model, _ = load_model(args)
    
    # Entraînement
    train_dataloader = DataLoader(train_dataset_tok, batch_size=args.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset_tok, batch_size=args.batch_size)
    
    train_model(args, model, tokenizer, train_dataloader, eval_dataloader, logger)
    
    logger.log("Training completed successfully!")
