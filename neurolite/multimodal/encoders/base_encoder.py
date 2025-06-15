"""
Module de base pour les encodeurs multimodaux pour le modèle principal.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from neurolite.Configs.config import ModelArchitectureConfig


class BaseEncoder(nn.Module):
    """
    Classe de base abstraite pour tous les encodeurs de modalités.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Chaque encodeur doit implémenter sa propre méthode forward.") 