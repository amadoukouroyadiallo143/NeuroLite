"""
Compresseur neuronal pour le tokenizer multimodal NeuroLite.

Ce module implémente un compresseur neuronal qui transforme
des représentations continues en tokens discrets compressés en utilisant
un auto-encodeur et un quantificateur vectoriel résiduel (RVQ).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any

from .quantizers import ResidualVQ


class NeuralCompressor(nn.Module):
    """
    Compresseur neuronal qui combine un auto-encodeur avec
    une quantification vectorielle résiduelle pour une tokenization efficace.
    Il s'agit du cœur de la tokenisation discrète.
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        num_quantizers: int = 8,
        codebook_size: int = 1024,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99
    ):
        """
        Initialise le compresseur neuronal.

        Args:
            input_dim: Dimension des vecteurs d'entrée (e.g., config.hidden_size).
            bottleneck_dim: Dimension du goulot d'étranglement de l'auto-encodeur.
            num_quantizers: Nombre de quantificateurs en cascade dans RVQ.
            codebook_size: Taille de chaque codebook dans RVQ.
            commitment_weight: Poids de la perte de commitment pour RVQ.
            ema_decay: Taux de decay pour les mises à jour EMA du codebook.
        """
        super().__init__()

        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        # Encodeur : projette l'entrée vers l'espace du goulot d'étranglement.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, bottleneck_dim)
        )

        # Quantificateur vectoriel résiduel : discrétise la représentation.
        self.vector_quantizer = ResidualVQ(
            dim=bottleneck_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            ema_decay=ema_decay
        )

        # Décodeur : reconstruit l'entrée originale à partir de la représentation quantifiée.
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Processus complet d'encodage, quantification et décodage.

        Args:
            x: Entrées à traiter [batch_size, ..., input_dim]

        Returns:
            Dictionnaire contenant:
            - 'z': Représentation latente avant quantification.
            - 'quantized': Représentation latente après quantification.
            - 'indices': Codes discrets (tokens) de chaque quantificateur.
            - 'reconstructed': Entrée reconstruite.
            - 'commitment_loss': Perte de commitment du RVQ.
        """
        # 1. Encoder les entrées vers l'espace latent
        z = self.encoder(x)

        # 2. Quantifier les encodages pour obtenir les codes discrets
        quantized, indices, commitment_loss = self.vector_quantizer(z)

        # 3. Décoder les représentations quantifiées pour la reconstruction
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
        Compresse les entrées en indices discrets (tokens) sans calculer les gradients.
        C'est la fonction à utiliser pour la tokenisation en inférence.

        Args:
            x: Entrées à compresser [batch_size, ..., input_dim]

        Returns:
            Liste des indices de quantification (un tenseur par quantificateur).
        """
        with torch.no_grad():
            z = self.encoder(x)
            # Seuls les indices sont retournés.
            _, indices, _ = self.vector_quantizer(z)
        return indices

    def decompress(self, indices: List[torch.Tensor]) -> torch.Tensor:
        """
        Décompresse les indices discrets en représentations continues.
        C'est la fonction à utiliser pour la détokenisation en inférence.

        Args:
            indices: Liste des indices de quantification.

        Returns:
            x_reconstructed: Entrées reconstruites [batch_size, ..., input_dim]
        """
        with torch.no_grad():
            # 1. Reconstruire les représentations quantifiées à partir des indices
            quantized = self.vector_quantizer.decode_indices(indices)

            # 2. Décoder les représentations quantifiées
            reconstructed = self.decoder(quantized)

        return reconstructed
