"""
Configurations prédéfinies pour différents modèles et tâches NeuroLite.

Ce module fournit des configurations prédéfinies pour différents cas d'utilisation,
du modèle léger (tiny) aux configurations plus grandes et spécialisées.
"""

from .config import (
    NeuroLiteConfig,
)

__all__ = [
    'NeuroLiteConfig',
]

# Création d'alias pour les configurations prédéfinies
# Ces alias permettent d'accéder facilement aux configurations courantes
tiny_config = NeuroLiteConfig.tiny
small_config = NeuroLiteConfig.small
base_config = NeuroLiteConfig.base
symbolic_config = NeuroLiteConfig.symbolic

# Ajout des alias à __all__
__all__.extend([
    'tiny_config',
    'small_config',
    'base_config',
    'symbolic_config',
])
