"""
Module de raisonnement avancé pour NeuroLite.

Ce module fournit des capacités de raisonnement symbolique, de planification 
et d'inférence pour des tâches nécessitant un haut niveau d'abstraction.

Classes principales :
- NeurosymbolicReasoner : Combine raisonnement neuronal et symbolique
- StructuredPlanner : Génère et évalue des plans d'action structurés
"""

from .reasoning import (
    NeurosymbolicReasoner,
    StructuredPlanner
)

__all__ = [
    'NeurosymbolicReasoner',
    'StructuredPlanner'
]
