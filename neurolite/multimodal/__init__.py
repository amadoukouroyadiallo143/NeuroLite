"""
Module de projection et d'attention multimodale pour NeuroLite.

Ce module permet d'intégrer et de fusionner des entrées de différentes modalités
(texte, image, audio, vidéo) dans un espace de représentation commun.
"""

from .multimodal import MultimodalProjection, CrossModalAttention

__all__ = [
    'MultimodalProjection',
    'CrossModalAttention'
]
