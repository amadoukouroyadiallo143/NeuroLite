"""
Module de mémoire pour NeuroLite.

Ce module fournit différentes implémentations de mémoires neuronales pour le stockage
et la récupération d'informations contextuelles, notamment :

- DifferentiableMemory : Mémoire associative différentiable
- ModernHopfieldLayer : Implémentation moderne des réseaux de Hopfield
- HierarchicalMemory : Architecture de mémoire hiérarchique à plusieurs niveaux
- VectorMemoryStore : Stockage persistant de vecteurs pour une récupération efficace
"""

from .hierarchical_memory import HierarchicalMemory, VectorMemoryStore
from .memory import DifferentiableMemory, ModernHopfieldLayer

__all__ = [
    'DifferentiableMemory',
    'ModernHopfieldLayer',
    'HierarchicalMemory',
    'VectorMemoryStore',
]
