"""
NeuroLite Base Tokenizer Classes
===============================

Classes de base et interfaces pour le système de tokenization universel.
Définit les protocoles et structures communes à tous les tokenizers.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)

class ModalityType(Enum):
    """Types de modalités supportées."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    STRUCTURED = "structured"
    UNKNOWN = "unknown"

class TokenizationStrategy(Enum):
    """Stratégies de tokenization disponibles."""
    # Texte
    BPE = "bpe"
    WORDPIECE = "wordpiece"
    SENTENCEPIECE = "sentencepiece"
    CHAR_LEVEL = "char_level"
    
    # Image
    PATCH = "patch"
    PIXEL = "pixel"
    FEATURE = "feature"
    VQ_VAE = "vq_vae"
    
    # Audio
    MEL_SPECTROGRAM = "mel_spectrogram"
    MFCC = "mfcc"
    WAVEFORM = "waveform"
    ENCODEC = "encodec"
    
    # Vidéo
    FRAME = "frame"
    MOTION = "motion"
    TEMPORAL = "temporal"
    CONV3D = "conv3d"
    
    # Structuré
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    GRAPH = "graph"

@dataclass
class TokenizerConfig:
    """Configuration universelle pour tous les tokenizers."""
    
    # Configuration générale
    vocab_size: int = 50000
    max_sequence_length: int = 4096
    padding_strategy: str = "max_length"  # "max_length", "longest", "do_not_pad"
    truncation: bool = True
    return_tensors: str = "torch"  # "torch", "numpy", "list"
    
    # Cache et performance
    cache_embeddings: bool = True
    cache_size_mb: float = 512.0
    enable_parallel: bool = True
    num_workers: int = 4
    
    # Monitoring et métriques
    enable_metrics: bool = True
    log_tokenization_time: bool = True
    
    # Configuration spécialisée par modalité
    text_config: Dict[str, Any] = field(default_factory=dict)
    image_config: Dict[str, Any] = field(default_factory=dict)
    audio_config: Dict[str, Any] = field(default_factory=dict)
    video_config: Dict[str, Any] = field(default_factory=dict)
    structured_config: Dict[str, Any] = field(default_factory=dict)
    
    def get_modality_config(self, modality: ModalityType) -> Dict[str, Any]:
        """Retourne la configuration spécifique pour une modalité."""
        config_map = {
            ModalityType.TEXT: self.text_config,
            ModalityType.IMAGE: self.image_config,
            ModalityType.AUDIO: self.audio_config,
            ModalityType.VIDEO: self.video_config,
            ModalityType.STRUCTURED: self.structured_config
        }
        return config_map.get(modality, {})

@dataclass
class TokenizationResult:
    """Résultat complet d'une tokenization."""
    
    # Données principales
    tokens: Union[torch.Tensor, np.ndarray, List[int]]
    embeddings: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    
    # Métadonnées
    modality: ModalityType = ModalityType.UNKNOWN
    strategy: TokenizationStrategy = None
    original_shape: Optional[Tuple[int, ...]] = None
    vocab_size: int = 0
    sequence_length: int = 0
    
    # Métriques de performance
    tokenization_time_ms: float = 0.0
    compression_ratio: float = 1.0
    memory_usage_mb: float = 0.0
    
    # Métadonnées additionnelles
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le résultat en dictionnaire."""
        return {
            'tokens': self.tokens,
            'embeddings': self.embeddings,
            'attention_mask': self.attention_mask,
            'modality': self.modality.value,
            'strategy': self.strategy.value if self.strategy else None,
            'original_shape': self.original_shape,
            'vocab_size': self.vocab_size,
            'sequence_length': self.sequence_length,
            'tokenization_time_ms': self.tokenization_time_ms,
            'compression_ratio': self.compression_ratio,
            'memory_usage_mb': self.memory_usage_mb,
            'metadata': self.metadata
        }

class BaseTokenizer(ABC):
    """Classe de base abstraite pour tous les tokenizers."""
    
    def __init__(self, config: TokenizerConfig, modality: ModalityType):
        """
        Initialise le tokenizer de base.
        
        Args:
            config: Configuration du tokenizer
            modality: Type de modalité traité
        """
        self.config = config
        self.modality = modality
        self.modality_config = config.get_modality_config(modality)
        
        # Statistiques
        self.total_tokenizations = 0
        self.total_time_ms = 0.0
        self.total_memory_mb = 0.0
        
        # Cache local si activé
        self._cache = {} if config.cache_embeddings else None
        
        logger.info(f"Initialisation {self.__class__.__name__} pour modalité {modality.value}")
    
    @abstractmethod
    def tokenize(self, data: Any, **kwargs) -> TokenizationResult:
        """
        Tokenise les données d'entrée.
        
        Args:
            data: Données à tokeniser
            **kwargs: Arguments additionnels spécifiques au tokenizer
            
        Returns:
            TokenizationResult: Résultat complet de la tokenization
        """
        pass
    
    @abstractmethod
    def detokenize(self, tokens: Union[torch.Tensor, np.ndarray, List[int]], **kwargs) -> Any:
        """
        Reconstruit les données originales à partir des tokens.
        
        Args:
            tokens: Tokens à détokeniser
            **kwargs: Arguments additionnels
            
        Returns:
            Données reconstruites
        """
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Retourne la taille du vocabulaire."""
        pass
    
    @abstractmethod
    def get_supported_strategies(self) -> List[TokenizationStrategy]:
        """Retourne les stratégies supportées par ce tokenizer."""
        pass
    
    def get_embeddings(self, tokens: Union[torch.Tensor, np.ndarray, List[int]]) -> torch.Tensor:
        """
        Convertit les tokens en embeddings (implémentation par défaut).
        
        Args:
            tokens: Tokens à convertir
            
        Returns:
            torch.Tensor: Embeddings correspondants
        """
        if isinstance(tokens, (list, np.ndarray)):
            tokens = torch.tensor(tokens)
        
        # Implémentation par défaut: embedding aléatoire pour démonstration
        vocab_size = self.get_vocab_size()
        embed_dim = self.modality_config.get('embed_dim', 512)
        
        # Créer une embedding layer simple
        if not hasattr(self, '_embedding_layer'):
            self._embedding_layer = torch.nn.Embedding(vocab_size, embed_dim)
        
        return self._embedding_layer(tokens)
    
    def _measure_performance(self, func):
        """Décorateur pour mesurer les performances."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Mise à jour des statistiques
            self.total_tokenizations += 1
            time_ms = (end_time - start_time) * 1000
            self.total_time_ms += time_ms
            memory_mb = (end_memory - start_memory) / 1024 / 1024
            self.total_memory_mb += memory_mb
            
            # Mise à jour du résultat avec les métriques
            if isinstance(result, TokenizationResult):
                result.tokenization_time_ms = time_ms
                result.memory_usage_mb = memory_mb
            
            return result
        return wrapper
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de performance du tokenizer."""
        avg_time = self.total_time_ms / max(1, self.total_tokenizations)
        avg_memory = self.total_memory_mb / max(1, self.total_tokenizations)
        
        return {
            'modality': self.modality.value,
            'total_tokenizations': self.total_tokenizations,
            'total_time_ms': self.total_time_ms,
            'average_time_ms': avg_time,
            'total_memory_mb': self.total_memory_mb,
            'average_memory_mb': avg_memory,
            'cache_enabled': self._cache is not None,
            'cache_size': len(self._cache) if self._cache else 0
        }
    
    def reset_statistics(self):
        """Remet à zéro les statistiques."""
        self.total_tokenizations = 0
        self.total_time_ms = 0.0
        self.total_memory_mb = 0.0
        if self._cache:
            self._cache.clear()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(modality={self.modality.value}, tokenizations={self.total_tokenizations})"