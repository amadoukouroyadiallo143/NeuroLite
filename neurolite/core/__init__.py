"""
NeuroLite Core - Modules principaux de l'AGI.
"""
# Modèle AGI principal (import sécurisé)
try:
    from .agi_model import NeuroLiteAGI, create_neurolite_agi, AGIMode
    AGI_MODEL_AVAILABLE = True
except ImportError:
    AGI_MODEL_AVAILABLE = False

# Exports conditionnels
__all__ = []

if AGI_MODEL_AVAILABLE:
    __all__.extend(['NeuroLiteAGI', 'create_neurolite_agi', 'AGIMode'])

# Version simplifiée pour éviter les erreurs d'import circulaire
# Les modules spécialisés sont importés directement depuis le __init__.py principal