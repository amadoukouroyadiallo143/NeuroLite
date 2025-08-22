"""
NeuroLite Tokenization Configuration
===================================

Configuration avancée pour le système de tokenization.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from .base_tokenizer import TokenizerConfig, ModalityType

@dataclass
class TextTokenizerConfig:
    """Configuration spécialisée pour TextTokenizer."""
    strategy: str = "bpe"
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    add_special_tokens: bool = True
    return_attention_mask: bool = True
    lowercase: bool = False
    remove_punctuation: bool = False

@dataclass
class ImageTokenizerConfig:
    """Configuration spécialisée pour ImageTokenizer."""
    strategy: str = "patch"
    patch_size: int = 16
    image_size: int = 224
    channels: int = 3
    embed_dim: int = 768
    normalize: bool = True
    resize_mode: str = "center_crop"

@dataclass
class AudioTokenizerConfig:
    """Configuration spécialisée pour AudioTokenizer."""
    strategy: str = "mel_spectrogram"
    sample_rate: int = 16000
    n_mels: int = 80
    hop_length: int = 160
    win_length: int = 400

@dataclass
class UniversalTokenizerConfig(TokenizerConfig):
    """Configuration complète pour UniversalTokenizer."""
    
    # Configurations spécialisées
    text_config: TextTokenizerConfig = field(default_factory=TextTokenizerConfig)
    image_config: ImageTokenizerConfig = field(default_factory=ImageTokenizerConfig)
    audio_config: AudioTokenizerConfig = field(default_factory=AudioTokenizerConfig)
    
    # Détection automatique
    auto_detect_modality: bool = True
    detection_confidence_threshold: float = 0.8
    
    # Performance
    enable_multiprocessing: bool = True
    max_workers: int = 4
    enable_gpu_acceleration: bool = True
    
    def __post_init__(self):
        """Post-traitement de la configuration."""
        # Convertir les configs spécialisées en dictionnaires
        self.text_config = self.text_config.__dict__ if hasattr(self.text_config, '__dict__') else self.text_config
        self.image_config = self.image_config.__dict__ if hasattr(self.image_config, '__dict__') else self.image_config
        self.audio_config = self.audio_config.__dict__ if hasattr(self.audio_config, '__dict__') else self.audio_config

def create_production_config() -> UniversalTokenizerConfig:
    """Crée une configuration optimisée pour la production."""
    return UniversalTokenizerConfig(
        vocab_size=50000,
        max_sequence_length=4096,
        cache_embeddings=True,
        cache_size_mb=1024.0,
        enable_parallel=True,
        num_workers=8,
        enable_metrics=True
    )

def create_development_config() -> UniversalTokenizerConfig:
    """Crée une configuration pour le développement."""
    return UniversalTokenizerConfig(
        vocab_size=10000,
        max_sequence_length=1024,
        cache_embeddings=True,
        cache_size_mb=256.0,
        enable_parallel=False,
        num_workers=1,
        enable_metrics=True
    )

def create_edge_config() -> UniversalTokenizerConfig:
    """Crée une configuration optimisée pour edge computing."""
    return UniversalTokenizerConfig(
        vocab_size=5000,
        max_sequence_length=512,
        cache_embeddings=False,
        enable_parallel=False,
        num_workers=1,
        enable_metrics=False
    )