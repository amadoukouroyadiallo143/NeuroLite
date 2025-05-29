#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'entraînement pour NeuroLite
"""

import os
import yaml
import torch
import logging
import argparse
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config_path):
        """Initialise le formateur avec la configuration"""
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(log_dir=self.config["training"]["logging_dir"])
        
        # Initialisation du modèle, des données, etc.
        self._setup()
    
    def _load_config(self, config_path):
        """Charge la configuration depuis un fichier YAML"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup(self):
        """Configure le modèle, les données et l'optimiseur"""
        # Ici, vous initialiserez votre modèle, vos données, etc.
        # Exemple simplifié :
        # self.model = NeuroLiteModel(config=self.config["model"]).to(self.device)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["training"]["learning_rate"])
        # self.scheduler = get_linear_schedule_with_warmup(...)
        pass
    
    def train_epoch(self, epoch):
        """Exécute une époque d'entraînement"""
        self.model.train()
        total_loss = 0
        
        for step, batch in enumerate(self.train_loader):
            # Ici, vous implémenterez la logique d'entraînement
            # Exemple :
            # inputs, labels = batch
            # outputs = self.model(inputs)
            # loss = outputs.loss
            # loss.backward()
            # self.optimizer.step()
            # self.scheduler.step()
            # self.optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if step % self.config["training"]["logging_steps"] == 0:
                logger.info(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
                self.writer.add_scalar('train/loss', loss.item(), global_step=step)
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def evaluate(self):
        """Évalue le modèle sur l'ensemble de validation"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Logique d'évaluation
                # outputs = self.model(...)
                # loss = outputs.loss
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """Sauvegarde un point de contrôle du modèle"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        output_dir = Path(self.config["training"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Si c'est le meilleur modèle, le sauvegarder séparément
        if is_best:
            best_path = output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    def train(self):
        """Boucle d'entraînement principale"""
        best_loss = float('inf')
        
        for epoch in range(self.config["training"]["num_train_epochs"]):
            logger.info(f"Début de l'époque {epoch+1}")
            
            # Entraînement
            train_loss = self.train_epoch(epoch)
            logger.info(f"Époque {epoch+1}, Perte d'entraînement: {train_loss:.4f}")
            
            # Évaluation
            val_loss = self.evaluate()
            logger.info(f"Époque {epoch+1}, Perte de validation: {val_loss:.4f}")
            
            # Sauvegarde
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                logger.info(f"Nouveau meilleur modèle avec une perte de {val_loss:.4f}")
            
            self.save_checkpoint(epoch, is_best=is_best)
            
            # Log TensorBoard
            self.writer.add_scalar('train/epoch_loss', train_loss, epoch)
            self.writer.add_scalar('val/loss', val_loss, epoch)
            
            # Log supplémentaire (ex: learning rate, poids, etc.)
            # self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], epoch)

def main():
    parser = argparse.ArgumentParser(description="Entraînement de NeuroLite")
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/base_config.yaml",
        help="Chemin vers le fichier de configuration YAML"
    )
    args = parser.parse_args()
    
    # Vérifier si le fichier de configuration existe
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Le fichier de configuration {args.config} n'existe pas.")
    
    # Démarrer l'entraînement
    trainer = Trainer(args.config)
    trainer.train()

if __name__ == "__main__":
    main()
