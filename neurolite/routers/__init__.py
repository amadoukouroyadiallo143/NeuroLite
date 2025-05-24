"""
Modules de routage pour diriger l'information entre les différents composants de NeuroLite.

Ce module implémente différentes stratégies de routage conditionnel pour activer
seulement les sous-modules pertinents selon l'entrée, économisant ainsi des calculs.
"""

from .routing import (
    MixtureOfExperts,
    SparseDispatcher,
    DynamicRoutingBlock
)

__all__ = [
    'MixtureOfExperts',
    'SparseDispatcher',
    'DynamicRoutingBlock',
]
