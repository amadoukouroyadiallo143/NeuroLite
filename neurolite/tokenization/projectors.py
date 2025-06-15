"""
Projecteurs et aligneurs multimodaux pour le tokenizer NeuroLite.

Ce module fournit des composants pour projeter et aligner des représentations
de différentes modalités dans un espace latent commun.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class CrossModalProjector(nn.Module):
    """
    Projecteur qui fusionne des représentations de différentes modalités
    dans un espace latent commun via une projection suivie d'une fusion
    pondérée par attention.
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        output_dim: int,
        dropout_rate: float = 0.1
    ):
        """
        Initialise le projecteur multimodal.
        
        Args:
            input_dims: Dictionnaire des dimensions d'entrée par modalité.
            output_dim: Dimension de sortie commune.
            dropout_rate: Taux de dropout.
        """
        super().__init__()
        self.output_dim = output_dim
        
        # Projecteurs linéaires pour chaque modalité
        self.projectors = nn.ModuleDict({
            mod: nn.Linear(dim, output_dim) for mod, dim in input_dims.items()
        })

        # Mécanisme d'attention simple pour pondérer l'importance des modalités
        self.attention = nn.Sequential(
            nn.Linear(output_dim, 1),
            nn.Tanh()
        )
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Projette et fusionne les caractéristiques multimodales.
        
        Args:
            features_dict: Dictionnaire des caractéristiques par modalité.
                           Shape: [batch_size, seq_len, input_dim]
            
        Returns:
            Représentation unifiée dans l'espace commun.
            Shape: [batch_size, seq_len, output_dim]
        """
        if not features_dict:
            raise ValueError("Le dictionnaire de caractéristiques est vide.")

        # Projeter chaque modalité individuellement
        projected_list = [
            self.projectors[mod](feat) for mod, feat in features_dict.items() if mod in self.projectors
        ]

        if not projected_list:
            raise ValueError("Aucune modalité du dictionnaire ne correspond à un projecteur initialisé.")

        # Si une seule modalité, la retourner directement après projection
        if len(projected_list) == 1:
            return projected_list[0]

        # Fusion par moyenne pondérée par attention
        # Stack pour le traitement par batch : [batch_size, seq_len, num_modalities, output_dim]
        stacked_features = torch.stack(projected_list, dim=-2)

        # Calculer les poids d'attention
        # Le score est basé sur le contenu de chaque modalité
        # [batch_size, seq_len, num_modalities, 1]
        attention_scores = self.attention(stacked_features)
        attention_weights = F.softmax(attention_scores, dim=-2)

        # Appliquer les poids et sommer
        # [batch_size, seq_len, output_dim]
        fused_features = torch.sum(stacked_features * attention_weights, dim=-2)

        return self.dropout(fused_features)
