"""
Module de base pour les décodeurs multimodaux.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from neurolite.Configs.config import MMDecoderConfigBase


class BaseDecoder(nn.Module):
    """
    Classe de base abstraite pour tous les décodeurs de modalités.
    """
    def __init__(self, config: MMDecoderConfigBase):
        super().__init__()
        self.config = config

    def forward(
        self,
        latent_representation: torch.Tensor,
        targets: Optional[Any] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Décode la représentation latente vers une sortie et calcule la perte.
        Doit retourner un dictionnaire contenant au moins 'logits' et 'loss'.
        """
        raise NotImplementedError("Chaque décodeur doit implémenter sa propre méthode forward.")

    def generate(self, latent_representation: torch.Tensor, **kwargs) -> Any:
        """
        Génère une sortie en mode inférence.
        """
        raise NotImplementedError("Chaque décodeur doit implémenter sa propre méthode generate.") 