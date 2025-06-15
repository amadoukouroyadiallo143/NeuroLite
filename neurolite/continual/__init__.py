"""
Module pour l'apprentissage continu dans NeuroLite.
Ce module fournit des fonctionnalités pour l'apprentissage continu et incrémental.
"""

from .continual import (
    ContinualAdapter,
    ReplayBuffer,
    ProgressiveCompressor
)
from .curriculum import CurriculumManager, text_length_scorer

__all__ = [
    'ContinualAdapter',
    'ReplayBuffer',
    'ProgressiveCompressor',
    'CurriculumManager',
    'text_length_scorer'
]
