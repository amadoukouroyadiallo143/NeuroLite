"""
NeuroLite: Architecture d'IA universelle légère pour appareils mobiles et embarqués
"""

# Modèles et configuration de base
from .model import NeuroLiteModel
from .config import NeuroLiteConfig

# Modules de mémoire
from .memory import DifferentiableMemory, ModernHopfieldLayer
from .hierarchical_memory import HierarchicalMemory, VectorMemoryStore

# Modules de traitement et projection
from .mixer import MLPBlock, MixerLayer, HyperMixer, FNetLayer
from .projection import MinHashBloomProjection, TokenizedMinHashProjection
from .routing import DynamicRoutingBlock, MixtureOfExperts

# Extensions AGI avancées
from .multimodal import MultimodalProjection, CrossModalAttention
from .continual import ContinualAdapter, ReplayBuffer, ProgressiveCompressor
from .reasoning import NeurosymbolicReasoner, StructuredPlanner
from .symbolic import NeuralSymbolicLayer

__version__ = "0.2.0"  # Mise à jour de la version avec capacités AGI
