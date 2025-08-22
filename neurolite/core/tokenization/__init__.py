"""
NeuroLite Universal Tokenization System v1.0
============================================

Système de tokenization universel pour tous types de données multimodales.
Architecture modulaire et extensible avec détection automatique des modalités.

Classes principales:
- UniversalTokenizer: Interface principale unifiée
- TokenizerRegistry: Gestionnaire central des tokenizers
- ModalityDetector: Détection automatique des types de données

Tokenizers spécialisés:
- TextTokenizer: Texte (BPE, WordPiece, SentencePiece)
- ImageTokenizer: Images (Patches, Pixels, VQ)
- AudioTokenizer: Audio (Mel-spectrograms, MFCC, Waveform)
- VideoTokenizer: Vidéo (Frames, Motion, Temporal)
- StructuredTokenizer: Données structurées (CSV, JSON, XML)

Auteur: NeuroLite Team
Version: 1.0.0
Date: Décembre 2024
"""

from .universal_tokenizer import UniversalTokenizer, get_universal_tokenizer
from .tokenizer_registry import TokenizerRegistry, register_tokenizer
from .modality_detectors import ModalityDetector, detect_modality
from .base_tokenizer import BaseTokenizer, TokenizerConfig, TokenizationResult, ModalityType

# Tokenizers spécialisés
from .tokenizers.text_tokenizer import TextTokenizer
from .tokenizers.image_tokenizer import ImageTokenizer
from .tokenizers.audio_tokenizer import AudioTokenizer
from .tokenizers.video_tokenizer import VideoTokenizer
from .tokenizers.structured_tokenizer import StructuredTokenizer

# Utilitaires
from .utils.caching import EmbeddingCache
from .utils.metrics import TokenizationMetrics
from .utils.optimization import TokenizerOptimizer

__version__ = "1.0.0"
__author__ = "NeuroLite Team"

__all__ = [
    # Classes principales
    'UniversalTokenizer',
    'TokenizerRegistry',
    'ModalityDetector',
    'BaseTokenizer',
    'TokenizerConfig',
    'TokenizationResult',
    'ModalityType',
    
    # Tokenizers spécialisés
    'TextTokenizer',
    'ImageTokenizer', 
    'AudioTokenizer',
    'VideoTokenizer',
    'StructuredTokenizer',
    
    # Utilitaires
    'EmbeddingCache',
    'TokenizationMetrics',
    'TokenizerOptimizer',
    
    # Fonctions utilitaires
    'register_tokenizer',
    'detect_modality',
]

# Configuration globale par défaut
DEFAULT_CONFIG = TokenizerConfig(
    vocab_size=50000,
    max_sequence_length=4096,
    padding_strategy="max_length",
    truncation=True,
    return_tensors="torch",
    cache_embeddings=True,
    enable_metrics=True
)

# Registry global par défaut
default_registry = TokenizerRegistry()

def get_universal_tokenizer(config: TokenizerConfig = None) -> UniversalTokenizer:
    """
    Retourne une instance du tokenizer universel configuré.
    
    Args:
        config: Configuration optionnelle, utilise DEFAULT_CONFIG si None
        
    Returns:
        UniversalTokenizer configuré et prêt à l'emploi
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    return UniversalTokenizer(config=config, registry=default_registry)

# Auto-registration des tokenizers par défaut
def _register_default_tokenizers():
    """Enregistre automatiquement les tokenizers par défaut."""
    try:
        default_registry.register('text', TextTokenizer)
        default_registry.register('image', ImageTokenizer)
        default_registry.register('audio', AudioTokenizer)
        default_registry.register('video', VideoTokenizer)
        default_registry.register('structured', StructuredTokenizer)
    except Exception as e:
        import logging
        logging.warning(f"Erreur lors de l'enregistrement des tokenizers par défaut: {e}")

# Enregistrement automatique au chargement du module
_register_default_tokenizers()