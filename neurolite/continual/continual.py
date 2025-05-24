"""
Module d'apprentissage continu pour NeuroLite.
Implémente des mécanismes permettant au modèle d'apprendre en continu
tout en évitant l'oubli catastrophique.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Optional, Union, Tuple, Any
import random

class ReplayBuffer:
    """
    Tampon de rejeu pour l'apprentissage continu.
    Stocke des exemples passés pour éviter l'oubli catastrophique.
    """
    
    def __init__(
        self, 
        capacity: int = 1000,
        strategy: str = "reservoir"
    ):
        self.capacity = capacity
        self.strategy = strategy
        self.buffer = []
        self.position = 0
        self.timestamps = []
        
    def add(self, experience: Any) -> None:
        """
        Ajoute une expérience au tampon selon la stratégie choisie.
        
        Args:
            experience: Expérience à mémoriser (entrée, sortie, etc.)
        """
        if self.strategy == "fifo":
            # First-In-First-Out: remplacer les plus anciens exemples
            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
                self.timestamps.append(time.time())
            else:
                self.buffer[self.position] = experience
                self.timestamps[self.position] = time.time()
                self.position = (self.position + 1) % self.capacity
                
        elif self.strategy == "reservoir":
            # Reservoir sampling: maintenir une distribution uniforme
            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
                self.timestamps.append(time.time())
            else:
                # Probabilité de remplacement décroissante avec le temps
                idx = random.randint(0, self.position)
                if idx < self.capacity:
                    self.buffer[idx] = experience
                    self.timestamps[idx] = time.time()
            self.position += 1
            
        elif self.strategy == "importance":
            # Basé sur l'importance: conserver les exemples difficiles
            # L'importance doit être incluse dans l'experience (e.g. loss)
            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
                self.timestamps.append(time.time())
            else:
                # Trouver l'exemple le moins important
                min_idx = np.argmin([exp.get('importance', 0) for exp in self.buffer])
                if experience.get('importance', 1.0) > self.buffer[min_idx].get('importance', 0):
                    self.buffer[min_idx] = experience
                    self.timestamps[min_idx] = time.time()
    
    def sample(self, batch_size: int) -> List[Any]:
        """
        Échantillonne un batch d'expériences du tampon.
        
        Args:
            batch_size: Nombre d'expériences à échantillonner
            
        Returns:
            Liste d'expériences échantillonnées
        """
        batch_size = min(batch_size, len(self.buffer))
        if batch_size == 0:
            return []
            
        return random.sample(self.buffer, batch_size)
        
    def __len__(self) -> int:
        return len(self.buffer)


class ContinualAdapter(nn.Module):
    """
    Adaptateur pour l'apprentissage continu.
    Module qui permet au modèle de s'adapter à de nouvelles données
    sans oublier les connaissances précédemment acquises.
    """
    
    def __init__(
        self,
        hidden_size: int,
        buffer_size: int = 100,
        adaptation_rate: float = 0.1,
        drift_threshold: float = 0.5,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.buffer_size = buffer_size
        self.adaptation_rate = adaptation_rate
        self.drift_threshold = drift_threshold
        
        # Mémoire d'expériences (replay buffer)
        self.buffer = ReplayBuffer(capacity=buffer_size)
        
        # Projecteur d'adaptation
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate)
        )
        
        # Détecteur de dérive conceptuelle
        self.drift_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Statistiques de distribution
        self.register_buffer('running_mean', torch.zeros(hidden_size))
        self.register_buffer('running_var', torch.ones(hidden_size))
        
    def forward(
        self, 
        x: torch.Tensor, 
        update_memory: bool = True
    ) -> torch.Tensor:
        """
        Adapte les représentations et maintient la mémoire épisodique.
        
        Args:
            x: Tensor d'entrée [batch_size, seq_len, hidden_size]
            update_memory: Si True, met à jour la mémoire interne
            
        Returns:
            Tensor adapté [batch_size, seq_len, hidden_size]
        """
        # Définir explicitement que nous sommes en mode évaluation par défaut
        # pour éviter des problèmes avec dropout, batch norm, etc.
        was_training = self.training
        self.eval()
        
        # Capturer le mode de gradient actuel
        grad_enabled = torch.is_grad_enabled()
        
        # Désactiver le calcul de gradient pour la partie "forward" standard
        with torch.set_grad_enabled(False):
            # Calculer la résiduelle
            residual = x
            
            # Appliquer l'adaptateur
            adapted = self.adapter(x)
            
            # Combiner avec l'entrée originale à un taux contrôlé
            adapted_x = residual + adapted * self.adaptation_rate
            
            # Mettre à jour les statistiques de distribution
            flat_x = x.reshape(-1, self.hidden_size)
            batch_mean = torch.mean(flat_x, dim=0)
            batch_var = torch.var(flat_x, dim=0, unbiased=False)
            
            # Mettre à jour la moyenne et variance glissantes
            momentum = 0.1
            self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
            self.running_var = (1 - momentum) * self.running_var + momentum * batch_var
            
            # Mettre à jour la mémoire si demandé
            if update_memory:
                # Calculer l'importance de l'exemple pour décider s'il sera mémorisé
                importance = self._compute_importance(x)
                
                # Mémoriser l'exemple si suffisamment important
                if importance > 0.1 or len(self.buffer) < 10:
                    # Stocker les tenseurs sans gradients pour économiser de la mémoire
                    current_sample = {
                        'input': x.clone(),  # pas besoin de detach car nous sommes déjà dans no_grad
                        'output': adapted_x.clone(),
                        'importance': importance,
                        'timestamp': time.time()
                    }
                    
                    self.buffer.add(current_sample)
                    
                    # Détecter si une adaptation plus profonde est nécessaire
                    if self._detect_drift(x):
                        # Activer le mode entraînement uniquement pour l'adaptation
                        self.train()
                        # Rétablir le réglage des gradients pour l'adaptation
                        with torch.set_grad_enabled(grad_enabled):
                            self._perform_adaptation()
                        # Retour au mode évaluation
                        self.eval()
        
        # Rétablir l'état d'entraînement précédent
        if was_training:
            self.train()
            
        return adapted_x
    
    def _compute_importance(self, x: torch.Tensor) -> float:
        """Calcule l'importance d'un exemple pour la mémoire"""
        # Calculer divergence par rapport à la distribution moyenne
        with torch.no_grad():
            flat_x = x.view(-1, self.hidden_size)
            normalized_x = (flat_x - self.running_mean) / (torch.sqrt(self.running_var) + 1e-5)
            importance = torch.mean(torch.abs(normalized_x)).item()
            
        return importance
    
    def _detect_drift(self, x: torch.Tensor) -> bool:
        """Détecte les changements significatifs dans la distribution d'entrée"""
        with torch.no_grad():
            drift_score = self.drift_detector(x.mean(dim=1)).mean().item()
            return drift_score > self.drift_threshold
    
    def _perform_adaptation(self) -> None:
        """
        Ajuste les poids de l'adaptateur en fonction des exemples mémorisés
        pour s'adapter à la distribution changeante.
        """
        # Assurer que nous sommes en mode d'entraînement
        self.train()
        
        # Ne pas faire d'adaptation si le buffer est vide
        if len(self.buffer) == 0:
            return
            
        # Créer un mini-batch à partir de la mémoire
        memory_samples = self.buffer.sample(min(10, len(self.buffer)))
        
        # Skip si pas assez d'échantillons
        if len(memory_samples) == 0:
            return
        
        try:
            # Extraire les données en s'assurant qu'elles sont des tenseurs
            # avec requires_grad=True
            device = next(self.parameters()).device
            
            # Convertir les échantillons en tenseurs
            inputs_list = []
            targets_list = []
            
            for sample in memory_samples:
                # Utiliser to() pour s'assurer que le tenseur est sur le bon device
                # et detach().clone() pour créer une nouvelle copie
                inp = sample['input'].detach().clone().to(device)
                tgt = sample['output'].detach().clone().to(device)
                
                # Assurer que les dimensions sont correctes (batch_size, seq_len, hidden_size)
                if len(inp.shape) == 2:
                    inp = inp.unsqueeze(0)
                if len(tgt.shape) == 2:
                    tgt = tgt.unsqueeze(0)
                
                inputs_list.append(inp)
                targets_list.append(tgt)
            
            # Concaténer les tenseurs et activer les gradients
            inputs = torch.cat(inputs_list, dim=0).requires_grad_(True)
            targets = torch.cat(targets_list, dim=0)  # pas besoin de gradients pour les cibles
            
            # Activer les gradients pour les paramètres de l'adaptateur uniquement
            for name, param in self.named_parameters():
                if 'adapter' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            # Mini-entraînement sur la mémoire
            optimizer = torch.optim.Adam([p for p in self.adapter.parameters() if p.requires_grad], lr=0.001)
            
            # Petit entraînement interne
            for _ in range(3):  # Quelques itérations seulement
                optimizer.zero_grad()
                
                # Forward pass avec l'adaptateur seul
                residual = inputs
                adapted = self.adapter(inputs)
                adapted = residual + adapted * self.adaptation_rate
                
                # Calculer la perte par rapport aux sorties mémorisées
                loss = F.mse_loss(adapted, targets)
                
                # Rétropropagation
                loss.backward()
                
                # Mettre à jour les poids de l'adaptateur
                optimizer.step()
                
            # Libérer la mémoire
            del inputs, targets, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except RuntimeError as e:
            print(f"Erreur lors de l'adaptation: {e}")
            # Continuer l'exécution même en cas d'erreur
            pass


class ProgressiveCompressor(nn.Module):
    """
    Module de compression progressive pour l'apprentissage continu.
    Permet de condenser les connaissances du modèle pour libérer
    de la capacité pour de nouvelles tâches.
    """
    
    def __init__(
        self,
        hidden_size: int,
        compression_ratio: float = 0.5,
        threshold: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.compression_ratio = compression_ratio
        self.threshold = threshold
        
        reduced_size = int(hidden_size * compression_ratio)
        
        # Encodeur pour compresser les représentations
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, reduced_size),
            nn.LayerNorm(reduced_size),
            nn.GELU()
        )
        
        # Décodeur pour reconstruire les représentations originales
        self.decoder = nn.Sequential(
            nn.Linear(reduced_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # Module de sélection pour identifier les connaissances importantes
        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compresse les représentations"""
        return self.encoder(x)
        
    def decompress(self, z: torch.Tensor) -> torch.Tensor:
        """Décompresse les représentations"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Effectue une compression sélective des connaissances.
        
        Args:
            x: Tensor d'entrée [batch_size, seq_len, hidden_size]
            
        Returns:
            Tuple de (représentations compressées, masque d'importance)
        """
        # Calculer l'importance de chaque dimension
        importance = self.importance_scorer(x)
        
        # Créer un masque binaire basé sur l'importance
        mask = (importance > self.threshold).float()
        
        # Compresser et décompresser
        z = self.compress(x)
        x_reconstructed = self.decompress(z)
        
        # Appliquer le masque: conserve seulement les dimensions importantes
        x_filtered = x * mask + x_reconstructed * (1 - mask)
        
        return x_filtered, mask
