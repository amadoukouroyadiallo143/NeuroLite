"""
Module d'apprentissage par contraste multimodal (MultiCLR) pour NeuroLite.
Étend les capacités de MultimodalProjection pour permettre l'alignement
des représentations entre différentes modalités sans supervision explicite,
en utilisant des techniques d'apprentissage contrastif.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np

from .multimodal import MultimodalProjection

class MultiCLR(nn.Module):
    """
    Contrastive Learning Representation pour données multimodales.
    Permet l'alignement des représentations entre différentes modalités
    en utilisant des techniques d'apprentissage contrastif.
    """
    
    def __init__(
        self,
        multimodal_projection: MultimodalProjection,
        temperature: float = 0.07,
        projection_dim: int = 128
    ):
        """
        Initialise le module MultiCLR.
        
        Args:
            multimodal_projection: Module MultimodalProjection existant
            temperature: Paramètre de température pour la similarité cosinus
            projection_dim: Dimension du vecteur de projection pour la contrastivité
        """
        super().__init__()
        
        self.multimodal_projection = multimodal_projection
        self.config = self.multimodal_projection.config
        self.temperature = temperature
        
        feature_dim = self.config.multimodal_hidden_dim
        
        self.projectors = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, projection_dim)
            )
            for name in self.multimodal_projection.encoders.encoders.keys()
        })
        
        self.register_buffer("mask", torch.eye(1, dtype=torch.bool))
        
    def get_individual_features(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Encode chaque modalité individuellement et la projette dans l'espace de fusion.
        """
        encoded = self.multimodal_projection.encoders(inputs)
        projected = {
            name: self.multimodal_projection.projections[name](features)
            for name, features in encoded.items()
        }
        return projected
    
    def forward(
        self,
        inputs: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calcule les projections pour chaque modalité pour la perte contrastive.
            
        Returns:
            - Perte contrastive totale.
            - Dictionnaire des projections normalisées pour chaque modalité.
        """
        individual_features = self.get_individual_features(inputs)
        
        projections = {
            name: F.normalize(self.projectors[name](feat), dim=-1)
            for name, feat in individual_features.items()
        }

        if len(projections) < 2:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device), projections

        total_loss = 0.0
        num_pairs = 0
        
        modalities = list(projections.keys())
        batch_size = projections[modalities[0]].shape[0]
        mask = self.mask.repeat(batch_size, batch_size)
        
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod1, mod2 = modalities[i], modalities[j]
                proj1, proj2 = projections[mod1], projections[mod2]

                sim_matrix = torch.matmul(proj1, proj2.T) / self.temperature
                sim_matrix_t = sim_matrix.T

                labels = torch.arange(batch_size, device=proj1.device)
                
                loss_i2j = F.cross_entropy(sim_matrix, labels)
                loss_j2i = F.cross_entropy(sim_matrix_t, labels)
                
                total_loss += (loss_i2j + loss_j2i) / 2.0
                num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else total_loss, projections


class MultiCLRHead(nn.Module):
    """
    Module de projection et prédiction pour l'apprentissage contrastif.
    Implémente la structure à double projection (projet + prédiction) du SimSiam/BYOL.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        use_predictor: bool = True,
        dropout_rate: float = 0.1
    ):
        """
        Initialisation du module MultiCLRHead
        
        Args:
            in_dim: Dimension d'entrée
            hidden_dim: Dimension cachée
            out_dim: Dimension de sortie
            use_predictor: Si True, ajoute un réseau de prédiction après la projection
            dropout_rate: Taux de dropout
        """
        super().__init__()
        
        # Projector comme dans SimCLR/BYOL/SimSiam
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        
        # Predictor optionnel comme dans BYOL/SimSiam
        self.use_predictor = use_predictor
        if use_predictor:
            self.predictor = nn.Sequential(
                nn.Linear(out_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, out_dim)
            )
        else:
            self.predictor = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Tensor d'entrée [batch_size, in_dim]
            
        Returns:
            Tuple contenant la projection et la prédiction
        """
        projection = self.projector(x)
        prediction = self.predictor(projection)
        
        return F.normalize(projection, dim=1), F.normalize(prediction, dim=1)
