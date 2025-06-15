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
    Un adaptateur léger pour l'apprentissage continu.
    Il est activé lorsque le modèle détecte une dérive de concept (concept drift).
    """
    def __init__(
        self,
        hidden_size: int,
        adapter_bottleneck_dim: int = 64, 
        task_embedding_dim: Optional[int] = None,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.adapter_bottleneck_dim = adapter_bottleneck_dim
        self.task_embedding_dim = task_embedding_dim
        
        # Le "goulot d'étranglement" de l'adaptateur
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, adapter_bottleneck_dim),
            nn.LayerNorm(adapter_bottleneck_dim),
            nn.GELU(),
            nn.Linear(adapter_bottleneck_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate)
        )
        
        # Module pour détecter si l'adaptateur doit être actif
        self.drift_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        self.task_embedding = None
        if task_embedding_dim:
            # Pour l'instant, on suppose un nombre fixe de tâches, mais cela pourrait être dynamique
            num_tasks = 10 
            self.task_embedding = nn.Embedding(num_tasks, task_embedding_dim)

    def forward(self, hidden_states: torch.Tensor, update: bool = False, task_id: Optional[int] = None) -> torch.Tensor:
        """
        Passe avant de l'adaptateur.
        
        Args:
            hidden_states: Les états cachés du modèle principal.
            update: Si True, met à jour l'adaptateur (phase d'entraînement).
            task_id: L'ID de la tâche actuelle pour utiliser le bon embedding de tâche.
            
        Returns:
            Les états cachés modifiés.
        """
        if not update:
            # En inférence, on pourrait utiliser le détecteur de dérive pour décider d'appliquer l'adaptateur
            drift_score = self.drift_detector(hidden_states.mean(dim=1)).mean() # Un score par batch
            if drift_score < 0.5: # Seuil arbitraire
                return hidden_states

        # Appliquer la transformation de l'adaptateur
        adapter_output = self.adapter(hidden_states)

        # Intégrer l'embedding de tâche si disponible
        if self.task_embedding is not None and task_id is not None:
            task_emb = self.task_embedding(torch.tensor([task_id], device=hidden_states.device))
            # L'embedding de tâche doit être broadcastable. On l'ajoute à la fin.
            # Il faudrait une projection pour que les dimensions correspondent.
            # Pour l'instant, on ignore cette partie complexe.
            pass

        # L'approche la plus simple est d'ajouter la sortie de l'adaptateur à l'entrée (connexion résiduelle)
        return hidden_states + adapter_output


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
