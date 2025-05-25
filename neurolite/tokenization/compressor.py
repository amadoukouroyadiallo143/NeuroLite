"""
Compresseur neuronal pour le tokenizer multimodal NeuroLite.

Ce module implémente un compresseur neuronal qui transforme
des représentations continues en tokens discrets compressés.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Any

from .quantizers import ResidualVQ


class NeuralCompressor(nn.Module):
    """
    Compresseur neuronal qui combine un auto-encodeur avec
    une quantification vectorielle pour une tokenization efficace.
    """
    
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        num_quantizers: int = 4,
        codebook_size: int = 8192,
        shared_codebook: bool = False,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99
    ):
        """
        Initialise le compresseur neuronal.
        
        Args:
            input_dim: Dimension des vecteurs d'entrée
            bottleneck_dim: Dimension du goulot d'étranglement
            num_quantizers: Nombre de quantificateurs en cascade
            codebook_size: Taille de chaque codebook
            shared_codebook: Si True, partage le même codebook entre tous les niveaux
            commitment_weight: Poids de la perte de commitment
            ema_decay: Taux de decay pour EMA
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        
        # Encodeur pour compression
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim)
        )
        
        # Quantificateur vectoriel résiduel
        self.vector_quantizer = ResidualVQ(
            dim=bottleneck_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            shared_codebook=shared_codebook,
            commitment_weight=commitment_weight,
            ema_decay=ema_decay
        )
        
        # Décodeur pour reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim)
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Encode et quantifie les entrées.
        
        Args:
            x: Entrées à encoder [batch_size, ..., input_dim]
            
        Returns:
            quantized: Représentations quantifiées [batch_size, ..., bottleneck_dim]
            indices: Liste des indices de quantification
            commitment_loss: Perte de commitment
        """
        # Encoder les entrées
        z = self.encoder(x)
        
        # Quantifier les encodages
        quantized, indices, commitment_loss = self.vector_quantizer(z)
        
        return quantized, indices, commitment_loss
    
    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        Décode les représentations quantifiées.
        
        Args:
            quantized: Représentations quantifiées [batch_size, ..., bottleneck_dim]
            
        Returns:
            reconstructed: Entrées reconstruites [batch_size, ..., input_dim]
        """
        return self.decoder(quantized)
    
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Processus complet d'encodage, quantification et décodage.
        
        Args:
            x: Entrées à traiter [batch_size, ..., input_dim]
            
        Returns:
            Dictionnaire contenant les résultats du processus
        """
        # Encoder les entrées
        z = self.encoder(x)
        
        # Quantifier les encodages
        quantized, indices, commitment_loss = self.vector_quantizer(z)
        
        # Décoder les représentations quantifiées
        reconstructed = self.decoder(quantized)
        
        return {
            'z': z,
            'quantized': quantized,
            'indices': indices,
            'reconstructed': reconstructed,
            'commitment_loss': commitment_loss
        }
    
    def compress(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Compresse les entrées en indices discrets.
        
        Args:
            x: Entrées à compresser [batch_size, ..., input_dim]
            
        Returns:
            indices: Liste des indices de quantification
        """
        with torch.no_grad():
            z = self.encoder(x)
            _, indices, _ = self.vector_quantizer(z)
        return indices
    
    def decompress(self, indices: List[torch.Tensor]) -> torch.Tensor:
        """
        Décompresse les indices en représentations continues.
        
        Args:
            indices: Liste des indices de quantification
            
        Returns:
            x: Entrées reconstruites [batch_size, ..., input_dim]
        """
        with torch.no_grad():
            # Reconstruire les représentations quantifiées à partir des indices
            quantized = self.vector_quantizer.decode_indices(indices)
            
            # Décoder les représentations quantifiées
            reconstructed = self.decoder(quantized)
        
        return reconstructed
