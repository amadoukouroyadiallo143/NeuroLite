#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script d'entraînement pour le tokenizer multimodal NeuroLite.

Ce script permet d'entraîner le tokenizer multimodal sur différents ensembles de données:
1. Entraînement du codebook sémantique pour l'extraction de caractéristiques générales
2. Entraînement du codebook détaillé pour la préservation des détails fins
3. Alignement cross-modal pour l'unification des représentations multimodales
4. Évaluation de la qualité des tokens sur différentes tâches

Exemples d'utilisation:
    python train_tokenizer.py --data_dir /path/to/data --output_dir /path/to/output
    python train_tokenizer.py --config configs/tokenizer_config.json --modalities text,image,audio
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

# Assurez-vous que le package neurolite est dans le PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurolite.tokenization import NeuroLiteTokenizer, TokenizerConfig
from neurolite.tokenization.quantizers import ResidualVQ, DualCodebookVQ

# Répertoires par défaut
DEFAULT_DATA_DIR = Path(os.path.join(os.path.dirname(__file__), '../data'))
DEFAULT_OUTPUT_DIR = Path(os.path.join(os.path.dirname(__file__), '../outputs/tokenizer_training'))
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration par défaut
DEFAULT_CONFIG = {
    "semantic_vocab_size": 8192,
    "detail_vocab_size": 32768,
    "hidden_size": 768,
    "dropout_rate": 0.1,
    "use_residual_vq": True,
    "num_residual_layers": 3,
    "commitment_weight": 0.25,
    "use_dual_codebook": True,
    "hierarchical_levels": 3,
    "use_context_modulation": True,
    "modality_alignment_weight": 0.5,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 10,
    "warmup_steps": 1000,
    "eval_steps": 500,
    "save_steps": 1000
}


class MultimodalDataset(Dataset):
    """
    Dataset pour l'entraînement multimodal du tokenizer.
    """
    def __init__(self, data_dir: str, modalities: List[str] = None):
        """
        Initialise le dataset multimodal.
        
        Args:
            data_dir: Répertoire contenant les données
            modalities: Liste des modalités à inclure
        """
        self.data_dir = Path(data_dir)
        self.modalities = modalities or ['text', 'image', 'audio', 'video', 'graph']
        
        # Simuler des données pour la démonstration
        # Dans une implémentation réelle, vous chargeriez les données à partir des fichiers
        self.num_samples = 100
        print(f"Initialisation du dataset avec {self.num_samples} échantillons pour les modalités: {', '.join(self.modalities)}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Charge un échantillon du dataset.
        
        Args:
            idx: Indice de l'échantillon
            
        Returns:
            Dictionnaire d'entrées multimodales
        """
        sample = {}
        
        # Générer des entrées simulées pour chaque modalité
        if 'text' in self.modalities:
            # Simuler des entrées textuelles
            text_options = [
                "L'architecture universelle d'IA combine des encodeurs modulaires et un noyau latent",
                "Une mémoire neurale multi-niveaux permet de traiter des séquences extrêmes",
                "Les flux d'information vont des encodeurs vers les latents et la mémoire",
                "Le routage adaptatif économise du calcul en activant les composants nécessaires",
                "Les attentions et SSM linéaires réduisent drastiquement le coût de calcul"
            ]
            sample['text'] = [text_options[idx % len(text_options)]]
        
        if 'image' in self.modalities:
            # Simuler des tenseurs d'image [C, H, W]
            sample['image'] = torch.randn(3, 224, 224)
        
        if 'audio' in self.modalities:
            # Simuler des tenseurs audio [C, T]
            sample_rate = 16000
            duration_sec = 3
            sample['audio'] = torch.randn(1, sample_rate * duration_sec)
        
        if 'video' in self.modalities:
            # Simuler des tenseurs vidéo [F, C, H, W]
            sample['video'] = torch.randn(8, 3, 224, 224)
        
        if 'graph' in self.modalities:
            # Simuler des données de graphe
            num_nodes = 16
            sample['graph'] = {
                'node_features': torch.randn(num_nodes, 64),
                'adjacency_matrix': torch.bernoulli(torch.ones(num_nodes, num_nodes) * 0.3),
                'node_mask': torch.ones(num_nodes)
            }
        
        return sample


def collate_multimodal_batch(batch):
    """
    Fonction de collate pour regrouper les échantillons multimodaux en batch.
    
    Args:
        batch: Liste d'échantillons
        
    Returns:
        Dictionnaire d'entrées multimodales en batch
    """
    collated = {}
    
    # Identifier les modalités présentes
    modalities = set()
    for sample in batch:
        modalities.update(sample.keys())
    
    # Traiter chaque modalité
    for modality in modalities:
        if modality == 'text':
            # Collecter les textes
            collated[modality] = []
            for sample in batch:
                if modality in sample:
                    collated[modality].extend(sample[modality])
        
        elif modality == 'graph':
            # Traitement spécial pour les graphes
            if any(modality in sample for sample in batch):
                # Dimensions
                batch_size = len(batch)
                sample_with_graph = next(sample for sample in batch if modality in sample)
                graph_sample = sample_with_graph[modality]
                num_nodes = graph_sample['node_features'].size(0)
                feat_dim = graph_sample['node_features'].size(1)
                
                # Initialiser les tenseurs
                node_features = torch.zeros(batch_size, num_nodes, feat_dim)
                adjacency = torch.zeros(batch_size, num_nodes, num_nodes)
                mask = torch.zeros(batch_size, num_nodes)
                
                # Remplir les tenseurs
                for i, sample in enumerate(batch):
                    if modality in sample:
                        graph = sample[modality]
                        node_features[i] = graph['node_features']
                        adjacency[i] = graph['adjacency_matrix']
                        mask[i] = graph['node_mask']
                
                collated[modality] = {
                    'node_features': node_features,
                    'adjacency_matrix': adjacency,
                    'node_mask': mask
                }
        
        else:
            # Pour les autres modalités (tenseurs)
            if any(modality in sample for sample in batch):
                # Trouver les dimensions
                sample_with_modality = next(sample for sample in batch if modality in sample)
                tensor_shape = sample_with_modality[modality].size()
                
                # Créer un tensor batch
                tensor_batch = []
                for sample in batch:
                    if modality in sample:
                        tensor_batch.append(sample[modality])
                    else:
                        # Zéros pour les échantillons manquants
                        tensor_batch.append(torch.zeros(tensor_shape))
                
                collated[modality] = torch.stack(tensor_batch)
    
    return collated


class TokenizerTrainer:
    """
    Classe pour l'entraînement du tokenizer multimodal.
    """
    def __init__(self, config: Dict[str, Any], output_dir: str):
        """
        Initialise le trainer.
        
        Args:
            config: Configuration pour le tokenizer et l'entraînement
            output_dir: Répertoire pour les sorties
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Créer le tokenizer
        tokenizer_config = {k: v for k, v in config.items() 
                            if k in TokenizerConfig.__annotations__}
        self.tokenizer_config = TokenizerConfig(**tokenizer_config)
        self.tokenizer = NeuroLiteTokenizer(self.tokenizer_config)
        
        # Paramètres d'entraînement
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.epochs = config.get('epochs', 10)
        self.warmup_steps = config.get('warmup_steps', 1000)
        self.eval_steps = config.get('eval_steps', 500)
        self.save_steps = config.get('save_steps', 1000)
        
        # Initialiser les métriques
        self.train_losses = []
        self.eval_losses = []
        self.current_step = 0
        
        # Initialiser les optimiseurs
        self.setup_optimizers()
    
    def setup_optimizers(self):
        """
        Configure les optimiseurs pour l'entraînement.
        """
        # Paramètres des modules VQ
        vq_params = []
        
        # Collecter les paramètres des modules VQ
        # Vérifier les noms d'attributs dans le tokenizer (compatibilité avec différentes versions)
        if hasattr(self.tokenizer, 'semantic_vq'):
            vq_params.extend(list(self.tokenizer.semantic_vq.parameters()))
        elif hasattr(self.tokenizer, 'semantic_codebook'):
            vq_params.extend(list(self.tokenizer.semantic_codebook.parameters()))
            
        if hasattr(self.tokenizer, 'detail_vq'):
            vq_params.extend(list(self.tokenizer.detail_vq.parameters()))
        elif hasattr(self.tokenizer, 'detail_codebook'):
            vq_params.extend(list(self.tokenizer.detail_codebook.parameters()))
            
        # Ajouter les paramètres du quantificateur résiduel s'il existe
        if hasattr(self.tokenizer, 'residual_quantizer'):
            vq_params.extend(list(self.tokenizer.residual_quantizer.parameters()))
            
        # Vérifier si nous avons un DualCodebookVQ
        if hasattr(self.tokenizer, 'dual_codebook_vq'):
            vq_params.extend(list(self.tokenizer.dual_codebook_vq.parameters()))
        
        # Paramètres des encodeurs/projections
        encoder_params = []
        
        # Vérifier d'abord la structure conventionnelle (encodeurs individuels)
        for encoder_name in ['text_encoder', 'image_encoder', 'audio_encoder', 
                            'video_encoder', 'graph_encoder']:
            if hasattr(self.tokenizer, encoder_name):
                encoder_params.extend(list(getattr(self.tokenizer, encoder_name).parameters()))
        
        # Vérifier l'architecture universelle avec projection multimodale
        if hasattr(self.tokenizer, 'multimodal_projection'):
            encoder_params.extend(list(self.tokenizer.multimodal_projection.parameters()))
            
            # Si la projection multimodale contient des encodeurs individuels
            if hasattr(self.tokenizer.multimodal_projection, 'encoders'):
                # Cas où les encodeurs sont stockés dans un ModuleDict ou un attribut 'encoders'
                for _, encoder in self.tokenizer.multimodal_projection.encoders.items():
                    encoder_params.extend(list(encoder.parameters()))
            
            # Vérification d'attributs encodeurs individuels dans multimodal_projection
            for encoder_name in ['text_encoder', 'image_encoder', 'audio_encoder', 
                                'video_encoder', 'graph_encoder']:
                if hasattr(self.tokenizer.multimodal_projection, encoder_name):
                    encoder_params.extend(list(getattr(self.tokenizer.multimodal_projection, encoder_name).parameters()))
        
        # Ajouter les autres modules importants du tokenizer
        for module_name in ['cross_modal_attention', 'post_attention_norm', 'post_attention_projection',
                            'codebook_refinement', 'output_projection']:
            if hasattr(self.tokenizer, module_name):
                encoder_params.extend(list(getattr(self.tokenizer, module_name).parameters()))
        
        # Optimiseurs séparés pour différentes parties du modèle
        self.vq_optimizer = optim.Adam(vq_params, lr=self.learning_rate)
        self.encoder_optimizer = optim.Adam(encoder_params, lr=self.learning_rate)
    
    def count_parameters(self, model):
        """
        Calcule le nombre total de paramètres entraînables du modèle.
        
        Args:
            model: Modèle PyTorch
            
        Returns:
            Nombre total de paramètres et dictionnaire avec le détail par module
        """
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Détail par module
        module_params = {}
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Module terminal (pas de sous-modules)
                num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                if num_params > 0:
                    module_params[name] = num_params
        
        # Regrouper par type de module
        grouped_params = {}
        for name, count in module_params.items():
            module_type = name.split('.')[-1]
            if any(key in name for key in ['encoder', 'semantic', 'detail', 'codebook', 'vq', 'quantizer']):
                # Extraire le type de module pour les modules importants
                for key in ['encoder', 'semantic', 'detail', 'codebook', 'vq', 'quantizer']:
                    if key in name:
                        module_type = key
                        break
            
            if module_type not in grouped_params:
                grouped_params[module_type] = 0
            grouped_params[module_type] += count
        
        return total_params, grouped_params
    
    def train(self, train_dataloader, eval_dataloader=None):
        """
        Entraîne le tokenizer.
        
        Args:
            train_dataloader: DataLoader pour l'entraînement
            eval_dataloader: DataLoader optionnel pour l'évaluation
        """
        # Calculer et afficher la taille du modèle
        total_params, grouped_params = self.count_parameters(self.tokenizer)
        print(f"\n=== Informations sur le modèle ===")
        print(f"Nombre total de paramètres: {total_params:,}")
        
        # Afficher les 5 principaux types de modules par nombre de paramètres
        print("\nRépartition des paramètres par type de module:")
        for module_type, count in sorted(grouped_params.items(), key=lambda x: x[1], reverse=True)[:5]:
            percentage = (count / total_params) * 100
            print(f"  - {module_type}: {count:,} ({percentage:.2f}%)")
        
        print(f"\nDébut de l'entraînement du tokenizer pour {self.epochs} époques")
        
        # Scheduler pour learning rate warmup
        total_steps = len(train_dataloader) * self.epochs
        
        # Boucle d'entraînement
        for epoch in range(self.epochs):
            print(f"\nÉpoque {epoch+1}/{self.epochs}")
            epoch_loss = self.train_epoch(train_dataloader)
            
            print(f"Perte moyenne de l'époque: {epoch_loss:.4f}")
            
            # Évaluation si un dataloader d'évaluation est fourni
            if eval_dataloader:
                eval_loss = self.evaluate(eval_dataloader)
                print(f"Perte d'évaluation: {eval_loss:.4f}")
                self.eval_losses.append(eval_loss)
            
            # Sauvegarder le modèle à la fin de chaque époque
            self.save_checkpoint(f"checkpoint_epoch_{epoch+1}")
        
        # Sauvegarder le modèle final
        self.save_checkpoint("final_model")
        
        # Visualiser les courbes d'apprentissage
        self.plot_learning_curves()
    
    def train_epoch(self, dataloader):
        """
        Entraîne le tokenizer pour une époque.
        
        Args:
            dataloader: DataLoader pour l'entraînement
            
        Returns:
            Perte moyenne de l'époque
        """
        self.tokenizer.train()
        epoch_loss = 0.0
        
        # Barre de progression
        progress_bar = tqdm(dataloader, desc="Entraînement")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Incrémenter le compteur de pas
            self.current_step += 1
            
            # Réinitialiser les gradients
            self.vq_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            
            # Tokeniser le batch (forward pass)
            tokens_output = self.tokenizer.tokenize(batch)
            
            # Calculer la perte
            # La perte est généralement calculée à l'intérieur du tokenizer et retournée
            # dans tokens_output, mais nous la recalculons ici pour plus de clarté
            
            # Vérifier si nous avons au moins une composante de perte disponible
            has_loss = False
            total_loss = 0.0
            
            # Perte de reconstruction VQ
            if 'vq_loss' in tokens_output:
                # Vérifier si c'est un tenseur ou un float
                vq_loss = tokens_output['vq_loss']
                if isinstance(vq_loss, float):
                    print("Avertissement: vq_loss est un float et non un tenseur")
                    vq_loss = torch.tensor(vq_loss, requires_grad=True, device=next(self.tokenizer.parameters()).device)
                total_loss += vq_loss
                has_loss = True
            
            # Perte d'alignement multimodal si présente
            if 'alignment_loss' in tokens_output:
                alignment_loss = tokens_output['alignment_loss']
                if isinstance(alignment_loss, float):
                    print("Avertissement: alignment_loss est un float et non un tenseur")
                    alignment_loss = torch.tensor(alignment_loss, requires_grad=True, device=next(self.tokenizer.parameters()).device)
                total_loss += alignment_loss * self.config.get('modality_alignment_weight', 0.5)
                has_loss = True
                
            # Si aucune perte n'est disponible, créer une perte factice pour débogage
            if not has_loss:
                print("Avertissement: Aucune composante de perte n'a été trouvée dans tokens_output")
                # Créer une perte factice basée sur un paramètre du modèle pour garder le graphe de calcul
                dummy_param = next(self.tokenizer.parameters())
                total_loss = torch.sum(dummy_param * 0.0)
            
            loss = total_loss
            
            # Rétropropagation
            loss.backward()
            
            # Mettre à jour les paramètres
            self.vq_optimizer.step()
            self.encoder_optimizer.step()
            
            # Accumuler la perte
            epoch_loss += loss.item()
            
            # Mettre à jour la barre de progression
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Évaluation périodique
            if self.eval_steps > 0 and self.current_step % self.eval_steps == 0:
                # Cette étape serait remplacée par un appel à self.evaluate() avec un dataloader d'évaluation
                self.train_losses.append(epoch_loss / (batch_idx + 1))
            
            # Sauvegarde périodique
            if self.save_steps > 0 and self.current_step % self.save_steps == 0:
                self.save_checkpoint(f"checkpoint_step_{self.current_step}")
        
        # Calculer la perte moyenne de l'époque
        epoch_loss /= len(dataloader)
        self.train_losses.append(epoch_loss)
        
        return epoch_loss
    
    def evaluate(self, dataloader):
        """
        Évalue le tokenizer.
        
        Args:
            dataloader: DataLoader pour l'évaluation
            
        Returns:
            Perte moyenne d'évaluation
        """
        self.tokenizer.eval()
        eval_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Évaluation"):
                # Tokeniser le batch
                tokens_output = self.tokenizer.tokenize(batch)
                
                # Calculer la perte
                loss = 0.0
                
                # Perte de reconstruction VQ
                if 'vq_loss' in tokens_output:
                    loss += tokens_output['vq_loss']
                
                # Perte d'alignement multimodal si présente
                if 'alignment_loss' in tokens_output:
                    loss += tokens_output['alignment_loss'] * self.config.get('modality_alignment_weight', 0.5)
                
                # Accumuler la perte
                eval_loss += loss.item()
        
        # Calculer la perte moyenne
        eval_loss /= len(dataloader)
        
        return eval_loss
    
    def save_checkpoint(self, name):
        """
        Sauvegarde un checkpoint du tokenizer.
        
        Args:
            name: Nom du checkpoint
        """
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Chemin du fichier de sauvegarde
        checkpoint_path = checkpoint_dir / f"{name}.pt"
        
        # Créer le dictionnaire de checkpoint
        checkpoint = {
            'tokenizer_config': self.tokenizer_config.__dict__,
            'model_state_dict': self.tokenizer.state_dict(),
            'vq_optimizer_state_dict': self.vq_optimizer.state_dict(),
            'encoder_optimizer_state_dict': self.encoder_optimizer.state_dict(),
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'current_step': self.current_step
        }
        
        # Sauvegarder le checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint sauvegardé: {checkpoint_path}")
        
        # Sauvegarder la configuration en JSON
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path):
        """
        Charge un checkpoint du tokenizer.
        
        Args:
            checkpoint_path: Chemin vers le fichier de checkpoint
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint non trouvé: {checkpoint_path}")
            return False
        
        # Charger le checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Recréer le tokenizer avec la configuration sauvegardée
        self.tokenizer_config = TokenizerConfig(**checkpoint['tokenizer_config'])
        self.tokenizer = NeuroLiteTokenizer(self.tokenizer_config)
        
        # Charger les états du modèle et des optimiseurs
        self.tokenizer.load_state_dict(checkpoint['model_state_dict'])
        self.vq_optimizer.load_state_dict(checkpoint['vq_optimizer_state_dict'])
        self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        
        # Restaurer les métriques
        self.train_losses = checkpoint['train_losses']
        self.eval_losses = checkpoint['eval_losses']
        self.current_step = checkpoint['current_step']
        
        print(f"Checkpoint chargé: {checkpoint_path}")
        return True
    
    def plot_learning_curves(self):
        """
        Trace les courbes d'apprentissage.
        """
        plt.figure(figsize=(12, 6))
        
        # Tracer la perte d'entraînement
        plt.plot(self.train_losses, label='Perte d\'entraînement')
        
        # Tracer la perte d'évaluation si disponible
        if self.eval_losses:
            # Interpoler les points d'évaluation pour correspondre à la longueur de train_losses
            eval_x = np.linspace(0, len(self.train_losses) - 1, len(self.eval_losses))
            plt.plot(eval_x, self.eval_losses, 'r-', label='Perte d\'évaluation')
        
        plt.xlabel('Époque')
        plt.ylabel('Perte')
        plt.title('Courbes d\'apprentissage du tokenizer')
        plt.legend()
        plt.grid(True)
        
        # Sauvegarder le graphique
        plt.savefig(self.output_dir / "learning_curves.png")
        plt.close()


def analyze_tokenizer(tokenizer, test_data, output_dir):
    """
    Analyse la qualité du tokenizer entraîné.
    
    Args:
        tokenizer: Tokenizer entraîné
        test_data: Données de test
        output_dir: Répertoire pour les sorties
    """
    print("\nAnalyse du tokenizer entraîné...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tokeniser les données de test
    tokenizer.eval()
    with torch.no_grad():
        tokens_output = tokenizer.tokenize(test_data)
    
    # Analyser la distribution des tokens
    if 'semantic_indices' in tokens_output and isinstance(tokens_output['semantic_indices'], torch.Tensor):
        semantic_indices = tokens_output['semantic_indices'].cpu().numpy()
        
        # Calculer les statistiques
        unique_indices = np.unique(semantic_indices)
        vocab_usage = len(unique_indices) / tokenizer.config.semantic_vocab_size
        
        print(f"Statistiques des tokens sémantiques:")
        print(f"  - Taille du vocabulaire: {tokenizer.config.semantic_vocab_size}")
        print(f"  - Nombre de tokens uniques utilisés: {len(unique_indices)}")
        print(f"  - Utilisation du vocabulaire: {vocab_usage:.2%}")
        
        # Visualiser la distribution des tokens
        plt.figure(figsize=(12, 6))
        plt.hist(semantic_indices.flatten(), bins=50)
        plt.title("Distribution des indices de tokens sémantiques")
        plt.xlabel("Indice de token")
        plt.ylabel("Fréquence")
        plt.savefig(output_dir / "token_distribution.png")
        plt.close()
    
    # Analyser la similarité cross-modale si disponible
    modality_keys = [key for key in tokens_output.keys() if key.endswith('_tokens') and isinstance(tokens_output[key], torch.Tensor)]
    
    if len(modality_keys) >= 2:
        print("\nAnalyse de la similarité cross-modale:")
        
        # Créer une matrice de similarité
        n_modalities = len(modality_keys)
        similarity_matrix = np.zeros((n_modalities, n_modalities))
        
        for i, key_i in enumerate(modality_keys):
            for j, key_j in enumerate(modality_keys):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Calculer la similarité cosinus moyenne
                    tokens_i = tokens_output[key_i]
                    tokens_j = tokens_output[key_j]
                    
                    # Normaliser
                    tokens_i_norm = F.normalize(tokens_i, p=2, dim=-1)
                    tokens_j_norm = F.normalize(tokens_j, p=2, dim=-1)
                    
                    # Calculer la similarité par batch et par position
                    sim = torch.bmm(tokens_i_norm, tokens_j_norm.transpose(1, 2))
                    similarity_matrix[i, j] = sim.mean().item()
        
        # Visualiser la matrice de similarité
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix, vmin=0, vmax=1, cmap='viridis')
        plt.colorbar(label="Similarité cosinus")
        
        # Définir les étiquettes
        labels = [key.replace('_tokens', '') for key in modality_keys]
        plt.xticks(range(n_modalities), labels, rotation=45)
        plt.yticks(range(n_modalities), labels)
        plt.title("Similarité cross-modale des représentations")
        
        # Ajouter les valeurs dans les cellules
        for i in range(n_modalities):
            for j in range(n_modalities):
                plt.text(j, i, f"{similarity_matrix[i, j]:.2f}", 
                        ha="center", va="center", 
                        color="white" if similarity_matrix[i, j] < 0.5 else "black")
        
        plt.tight_layout()
        plt.savefig(output_dir / "cross_modal_similarity.png")
        plt.show()
    
    # Sauvegarder les résultats d'analyse
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tokenizer_config": tokenizer.config.__dict__,
    }
    
    if 'semantic_indices' in tokens_output:
        results.update({
            "vocab_size": tokenizer.config.semantic_vocab_size,
            "unique_tokens": len(unique_indices),
            "vocab_usage": vocab_usage
        })
    
    # Sauvegarder en JSON
    with open(output_dir / "analysis_results.json", 'w', encoding='utf-8') as f:
        # Convertir les objets non sérialisables
        clean_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                clean_results[key] = {k: v for k, v in value.items() 
                                    if not isinstance(v, (torch.Tensor, nn.Module))}
            elif not isinstance(value, (torch.Tensor, nn.Module)):
                clean_results[key] = value
        
        json.dump(clean_results, f, indent=2)


def main():
    """
    Fonction principale du script.
    """
    parser = argparse.ArgumentParser(description="Entraînement du tokenizer multimodal NeuroLite")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help="Répertoire contenant les données d'entraînement")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Répertoire pour les sorties")
    parser.add_argument("--config", type=str, default=None,
                        help="Chemin vers un fichier de configuration JSON")
    parser.add_argument("--modalities", type=str, default=None,
                        help="Liste de modalités à inclure, séparées par des virgules")
    parser.add_argument("--resume", type=str, default=None,
                        help="Chemin vers un checkpoint pour reprendre l'entraînement")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Seulement analyser un tokenizer existant sans entraînement")
    
    args = parser.parse_args()
    
    # Traiter les modalités si spécifiées
    modalities = None
    if args.modalities:
        modalities = args.modalities.split(',')
    
    # Charger la configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG
    
    # Créer les datasets
    train_dataset = MultimodalDataset(args.data_dir, modalities)
    
    # Pour la démonstration, utiliser le même dataset pour l'évaluation
    # Dans une implémentation réelle, vous auriez un dataset d'évaluation séparé
    eval_dataset = MultimodalDataset(args.data_dir, modalities)
    
    # Créer les dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        collate_fn=collate_multimodal_batch
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        collate_fn=collate_multimodal_batch
    )
    
    # Créer le trainer
    trainer = TokenizerTrainer(config, args.output_dir)
    
    # Reprendre l'entraînement si spécifié
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Seulement analyser si demandé
    if args.analyze_only:
        # Obtenir un batch de test
        test_batch = next(iter(eval_dataloader))
        analyze_tokenizer(trainer.tokenizer, test_batch, args.output_dir)
    else:
        # Entraîner le tokenizer
        trainer.train(train_dataloader, eval_dataloader)
        
        # Analyser le tokenizer entraîné
        test_batch = next(iter(eval_dataloader))
        analyze_tokenizer(trainer.tokenizer, test_batch, args.output_dir)


if __name__ == "__main__":
    main()
