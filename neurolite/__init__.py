"""
NeuroLite - Une architecture d'IA modulaire et évolutive

Ce module implémente une architecture d'IA inspirée des dernières avancées en modèles de séquence,
avec une attention particulière portée à l'efficacité et à la polyvalence.
"""

# Import des composants principaux
from .core.model import NeuroLiteModel
from .Configs.config import NeuroLiteConfig
from .reasoning import NeurosymbolicReasoner, StructuredPlanner

# Version du package
__version__ = "0.1.0"

# Classes et fonctions à exposer au niveau du package
__all__ = [
    'NeuroLiteModel',
    'NeuroLiteConfig',
    'NeurosymbolicReasoner',
    'StructuredPlanner',
]
