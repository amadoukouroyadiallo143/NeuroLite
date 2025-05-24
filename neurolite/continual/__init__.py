"""
Module pour l'apprentissage continu dans NeuroLite.
Ce module fournit des fonctionnalités pour l'apprentissage continu et incrémental.
"""

from .continual import (
    ContinualAdapter,
    ReplayBuffer,
    ProgressiveCompressor
)

__all__ = [
    'ContinualAdapter',
    'ReplayBuffer',
    'ProgressiveCompressor'
]
