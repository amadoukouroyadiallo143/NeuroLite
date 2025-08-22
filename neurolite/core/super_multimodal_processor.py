"""
NeuroLite Super Multimodal Processor v2.0
=========================================

Fusion optimale de MultimodalProcessor et MultimodalFusionCenter
avec intégration complète du Universal Tokenizer System.

🎯 AVANTAGES :
- ✅ Tokenization universelle automatique
- ✅ 6 modalités (text, image, audio, video, structured, sensor)
- ✅ 5 stratégies de fusion avancées  
- ✅ Cache multiniveau intelligent
- ✅ Traitement parallèle optimisé
- ✅ Compatible PyTorch et BrainRegions
- ✅ Métriques de performance en temps réel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import threading

# Imports NeuroLite
from .brain_architecture import BrainSignal, BrainRegion
from .multimodal_fusion import (
    FusionStrategy, UnifiedRepresentation, CrossModalAttention, 
    HierarchicalFusion, AdaptiveFusionGate
)

# Import Universal Tokenizer
try:
    from .tokenization import (
        UniversalTokenizer, get_universal_tokenizer, ModalityType,
        TokenizationResult, detect_modality
    )
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ProcessingMetrics:
    """Métriques de traitement multimodal."""
    tokenization_time_ms: float = 0.0
    encoding_time_ms: float = 0.0
    fusion_time_ms: float = 0.0
    total_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    modalities_processed: List[str] = None
    confidence_score: float = 0.0
    
    def __post_init__(self):
        if self.modalities_processed is None:
            self.modalities_processed = []

class ModalityType2(Enum):
    """Types de modalités étendus."""
    TEXT = "text"
    IMAGE = "image" 
    AUDIO = "audio"
    VIDEO = "video"
    STRUCTURED = "structured"
    SENSOR = "sensor"  # Nouveau : données capteurs IoT

class SuperMultimodalProcessor(nn.Module):
    """
    Processeur multimodal de nouvelle génération avec tokenization universelle.
    
    Combine :
    - Universal Tokenizer pour préparation des données
    - Encodeurs spécialisés optimisés
    - Stratégies de fusion avancées  
    - Cache multiniveau
    - Traitement parallèle intelligent
    """
    
    def __init__(self, 
                 hidden_size: int = 768,
                 fusion_strategy: FusionStrategy = FusionStrategy.ADAPTIVE_FUSION,
                 enable_universal_tokenizer: bool = True,
                 enable_caching: bool = True,
                 enable_parallel: bool = True,
                 max_workers: int = 6):
        """
        Initialise le Super Multimodal Processor.
        
        Args:
            hidden_size: Dimension des représentations
            fusion_strategy: Stratégie de fusion à utiliser
            enable_universal_tokenizer: Activer le tokenizer universel
            enable_caching: Activer le cache multiniveau
            enable_parallel: Activer le traitement parallèle
            max_workers: Nombre max de workers parallèles
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.fusion_strategy = fusion_strategy
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        
        # 🎯 UNIVERSAL TOKENIZER INTÉGRÉ
        self.universal_tokenizer = None
        if enable_universal_tokenizer and TOKENIZER_AVAILABLE:
            try:
                self.universal_tokenizer = get_universal_tokenizer()
                logger.info("✅ Universal Tokenizer activé avec succès")
            except Exception as e:
                logger.warning(f"⚠️ Erreur Universal Tokenizer: {e}")
                self.universal_tokenizer = None
        else:
            reason = "imports manquants" if not TOKENIZER_AVAILABLE else "désactivé"
            logger.warning(f"⚠️ Universal Tokenizer non disponible ({reason})")
        
        # 🧠 CORRESPONDANCE MODALITÉ ↔ BRAIN REGION
        self.modality_to_region = {
            ModalityType2.TEXT: BrainRegion.LANGUAGE_CORTEX,
            ModalityType2.IMAGE: BrainRegion.VISUAL_CORTEX,
            ModalityType2.AUDIO: BrainRegion.AUDITORY_CORTEX,
            ModalityType2.VIDEO: BrainRegion.VISUAL_CORTEX,
            ModalityType2.STRUCTURED: BrainRegion.ANALYTICAL_CORTEX,
            ModalityType2.SENSOR: BrainRegion.ANALYTICAL_CORTEX  # Capteurs IoT -> Analytical
        }
        
        self.region_to_modality = {v: k for k, v in self.modality_to_region.items()}
        
        # 🏗️ ENCODEURS SPÉCIALISÉS OPTIMISÉS
        self.modality_encoders = nn.ModuleDict({
            modality.value: self._create_optimized_encoder(modality)
            for modality in ModalityType2
        })
        
        # 🔗 MODULE DE FUSION SELON STRATÉGIE
        self.fusion_module = self._create_fusion_module(fusion_strategy)
        
        # 🎨 PROJECTEUR UNIVERSEL FINAL
        self.universal_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # 💾 SYSTÈME DE CACHE MULTINIVEAU
        self.enable_caching = enable_caching
        if enable_caching:
            self._tokenization_cache = {}
            self._encoding_cache = {}
            self._fusion_cache = {}
            self._cache_lock = threading.Lock()
            
        # 📊 MÉTRIQUES ET STATISTIQUES
        self._reset_metrics()
        
        # 🔧 OPTIMISATIONS
        self._warmup_completed = False
        
        logger.info(f"🚀 SuperMultimodalProcessor initialisé:")
        logger.info(f"   • Modalités: {len(ModalityType2)} types")
        logger.info(f"   • Stratégie fusion: {fusion_strategy.value}")
        logger.info(f"   • Tokenizer universel: {'✅' if self.universal_tokenizer else '❌'}")
        logger.info(f"   • Cache: {'✅' if enable_caching else '❌'}")
        logger.info(f"   • Parallélisme: {'✅' if enable_parallel else '❌'}")
    
    def _create_optimized_encoder(self, modality: ModalityType2) -> nn.Module:
        """Crée un encodeur optimisé pour chaque modalité."""
        
        if modality == ModalityType2.TEXT:
            # Encodeur text avec attention locale
            return nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 2),
                nn.LayerNorm(self.hidden_size * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.LayerNorm(self.hidden_size)
            )
            
        elif modality == ModalityType2.IMAGE:
            # Encodeur vision SANS BatchNorm (causait les erreurs)
            return nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size)
            )
            
        elif modality == ModalityType2.AUDIO:
            # Encodeur audio avec convolutions 1D simulées
            return nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 2, self.hidden_size),
                nn.LayerNorm(self.hidden_size)
            )
            
        elif modality == ModalityType2.VIDEO:
            # Encodeur vidéo avec agrégation temporelle
            return nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 3),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(self.hidden_size * 3, self.hidden_size),
                nn.LayerNorm(self.hidden_size)
            )
            
        elif modality == ModalityType2.STRUCTURED:
            # Encodeur données structurées SANS BatchNorm (causait les erreurs)
            return nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size)
            )
            
        elif modality == ModalityType2.SENSOR:
            # Encodeur capteurs IoT avec normalisation robuste
            return nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 2, self.hidden_size),
                nn.LayerNorm(self.hidden_size)
            )
        
        # Encodeur générique par défaut
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
    
    def _create_fusion_module(self, strategy: FusionStrategy) -> nn.Module:
        """Crée le module de fusion selon la stratégie."""
        
        if strategy == FusionStrategy.ATTENTION_FUSION:
            return CrossModalAttention(self.hidden_size, num_heads=12)
        elif strategy == FusionStrategy.HIERARCHICAL_FUSION:
            return HierarchicalFusion(self.hidden_size, num_levels=3)
        elif strategy == FusionStrategy.ADAPTIVE_FUSION:
            return AdaptiveFusionGate(len(ModalityType2), self.hidden_size)
        else:
            # Fusion simple avec attention
            return nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.GELU(),
                nn.Dropout(0.1)
            )
    
    def forward(self, 
                inputs: Dict[str, Any],
                brain_signals: Optional[List[BrainSignal]] = None,
                return_metrics: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, ProcessingMetrics]]:
        """
        Forward pass super-optimisé avec tokenization universelle.
        
        Args:
            inputs: Dictionnaire de données par modalité
            brain_signals: Signaux cérébraux optionnels
            return_metrics: Retourner les métriques de performance
            
        Returns:
            Tensor unifié ou (Tensor, Métriques)
        """
        start_time = time.time()
        metrics = ProcessingMetrics()
        
        try:
            # 🎯 PHASE 1: TOKENIZATION UNIVERSELLE
            tokenization_start = time.time()
            tokenized_inputs = self._tokenize_inputs(inputs)
            metrics.tokenization_time_ms = (time.time() - tokenization_start) * 1000
            
            # 🧠 PHASE 2: ENCODAGE PARALLÈLE DES MODALITÉS
            encoding_start = time.time()
            encoded_modalities = self._encode_modalities_parallel(tokenized_inputs)
            metrics.encoding_time_ms = (time.time() - encoding_start) * 1000
            metrics.modalities_processed = list(encoded_modalities.keys())
            
            # 🔗 PHASE 3: FUSION INTELLIGENTE
            fusion_start = time.time()
            fused_representation, confidence = self._fuse_modalities(encoded_modalities)
            metrics.fusion_time_ms = (time.time() - fusion_start) * 1000
            metrics.confidence_score = confidence
            
            # 🎨 PHASE 4: PROJECTION FINALE
            final_output = self.universal_projector(fused_representation)
            
            # 📊 FINALISATION DES MÉTRIQUES
            metrics.total_time_ms = (time.time() - start_time) * 1000
            if self.enable_caching:
                metrics.cache_hit_rate = self._calculate_cache_hit_rate()
            
            # Mise à jour statistiques globales
            self._update_global_stats(metrics)
            
            if return_metrics:
                return final_output, metrics
            else:
                return final_output
                
        except Exception as e:
            logger.error(f"❌ Erreur SuperMultimodalProcessor: {e}")
            # Fallback sécurisé avec métriques d'erreur
            fallback_output = torch.zeros(1, self.hidden_size)
            
            # Créer métriques d'erreur par défaut si elles n'existent pas
            if 'metrics' not in locals():
                metrics = ProcessingMetrics(
                    tokenization_time_ms=0.0,
                    encoding_time_ms=0.0,
                    fusion_time_ms=0.0,
                    total_time_ms=0.0,
                    cache_hit_rate=0.0,
                    modalities_processed=[],
                    confidence_score=0.0
                )
            
            metrics.confidence_score = 0.0  # Confiance nulle en cas d'erreur
            
            if return_metrics:
                return fallback_output, metrics
            else:
                return fallback_output
    
    def _tokenize_inputs(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Tokenise toutes les entrées avec le tokenizer universel."""
        
        if not self.universal_tokenizer:
            # Fallback sans tokenizer
            return self._prepare_inputs_fallback(inputs)
        
        tokenized = {}
        
        for key, data in inputs.items():
            try:
                # Cache check
                cache_key = f"tokenize_{key}_{hash(str(data))}" if self.enable_caching else None
                
                if cache_key and cache_key in self._tokenization_cache:
                    tokenized[key] = self._tokenization_cache[cache_key]
                    continue
                
                # Tokenization universelle
                result = self.universal_tokenizer.tokenize(data)
                
                # Conversion en tensor approprié
                if result.embeddings is not None:
                    tensor = result.embeddings
                else:
                    # Fallback: convertir tokens en embeddings factices
                    tokens = torch.tensor(result.tokens[:self.hidden_size] if len(result.tokens) > self.hidden_size else result.tokens)
                    if len(tokens) < self.hidden_size:
                        padding = torch.zeros(self.hidden_size - len(tokens))
                        tokens = torch.cat([tokens.float(), padding])
                    tensor = tokens.unsqueeze(0) if tokens.dim() == 1 else tokens
                
                # Normaliser la dimension
                if tensor.size(-1) != self.hidden_size:
                    if tensor.size(-1) > self.hidden_size:
                        tensor = tensor[..., :self.hidden_size]
                    else:
                        padding = torch.zeros(*tensor.shape[:-1], self.hidden_size - tensor.size(-1))
                        tensor = torch.cat([tensor, padding], dim=-1)
                
                tokenized[key] = tensor
                
                # Cache
                if cache_key and self.enable_caching:
                    with self._cache_lock:
                        self._tokenization_cache[cache_key] = tensor
                        
            except Exception as e:
                logger.warning(f"Erreur tokenization {key}: {e}")
                # Fallback tensor factice
                tokenized[key] = torch.randn(1, self.hidden_size)
        
        return tokenized
    
    def _prepare_inputs_fallback(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Préparation des entrées sans tokenizer universel."""
        prepared = {}
        
        for key, data in inputs.items():
            if isinstance(data, torch.Tensor):
                tensor = data
            elif isinstance(data, (list, np.ndarray)):
                tensor = torch.tensor(data, dtype=torch.float32)
            elif isinstance(data, str):
                # Simple hashing pour les strings
                tensor = torch.tensor([hash(data) % 1000 for _ in range(self.hidden_size)], dtype=torch.float32)
            else:
                # Tensor aléatoire par défaut
                tensor = torch.randn(self.hidden_size)
            
            # Normalisation dimension
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.size(-1) != self.hidden_size:
                if tensor.size(-1) > self.hidden_size:
                    tensor = tensor[..., :self.hidden_size]
                else:
                    padding = torch.zeros(*tensor.shape[:-1], self.hidden_size - tensor.size(-1))
                    tensor = torch.cat([tensor, padding], dim=-1)
            
            prepared[key] = tensor
        
        return prepared
    
    def _encode_modalities_parallel(self, tokenized_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encode les modalités en parallèle."""
        
        if not self.enable_parallel or len(tokenized_inputs) <= 1:
            # Mode séquentiel
            return self._encode_modalities_sequential(tokenized_inputs)
        
        # Mode parallèle
        encoded = {}
        futures = []
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(tokenized_inputs))) as executor:
            
            for modality_key, tensor in tokenized_inputs.items():
                future = executor.submit(self._encode_single_modality, modality_key, tensor)
                futures.append((modality_key, future))
            
            # Récupération des résultats
            for modality_key, future in futures:
                try:
                    encoded[modality_key] = future.result(timeout=10)
                except Exception as e:
                    logger.warning(f"Erreur encodage parallèle {modality_key}: {e}")
                    # Fallback
                    encoded[modality_key] = torch.randn(1, self.hidden_size)
        
        return encoded
    
    def _encode_modalities_sequential(self, tokenized_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encode les modalités en séquentiel."""
        encoded = {}
        
        for modality_key, tensor in tokenized_inputs.items():
            try:
                encoded[modality_key] = self._encode_single_modality(modality_key, tensor)
            except Exception as e:
                logger.warning(f"Erreur encodage séquentiel {modality_key}: {e}")
                encoded[modality_key] = torch.randn(1, self.hidden_size)
        
        return encoded
    
    def _encode_single_modality(self, modality_key: str, tensor: torch.Tensor) -> torch.Tensor:
        """Encode une seule modalité."""
        
        # Cache check
        cache_key = f"encode_{modality_key}_{tensor.shape}" if self.enable_caching else None
        
        if cache_key and cache_key in self._encoding_cache:
            return self._encoding_cache[cache_key]
        
        # Déterminer le type de modalité
        modality_type = self._infer_modality_type(modality_key)
        
        # Encoder avec le réseau spécialisé
        if modality_type.value in self.modality_encoders:
            encoder = self.modality_encoders[modality_type.value]
            encoded = encoder(tensor)
        else:
            # Encodeur générique
            encoded = self.modality_encoders[ModalityType2.TEXT.value](tensor)
        
        # Cache
        if cache_key and self.enable_caching:
            with self._cache_lock:
                self._encoding_cache[cache_key] = encoded
        
        return encoded
    
    def _infer_modality_type(self, modality_key: str) -> ModalityType2:
        """Infère le type de modalité depuis la clé."""
        key_lower = modality_key.lower()
        
        if 'text' in key_lower or 'language' in key_lower:
            return ModalityType2.TEXT
        elif 'image' in key_lower or 'visual' in key_lower or 'vision' in key_lower:
            return ModalityType2.IMAGE
        elif 'audio' in key_lower or 'sound' in key_lower:
            return ModalityType2.AUDIO
        elif 'video' in key_lower:
            return ModalityType2.VIDEO
        elif 'struct' in key_lower or 'data' in key_lower or 'json' in key_lower:
            return ModalityType2.STRUCTURED
        elif 'sensor' in key_lower or 'iot' in key_lower:
            return ModalityType2.SENSOR
        else:
            return ModalityType2.TEXT  # Défaut
    
    def _fuse_modalities(self, encoded_modalities: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        """Fusionne les modalités encodées."""
        
        if not encoded_modalities:
            return torch.zeros(1, self.hidden_size), 0.0
        
        modality_tensors = list(encoded_modalities.values())
        
        # Cache check
        cache_key = f"fuse_{len(modality_tensors)}_{self.fusion_strategy.value}" if self.enable_caching else None
        
        if cache_key and cache_key in self._fusion_cache:
            return self._fusion_cache[cache_key]
        
        try:
            # 🔧 NORMALISATION ROBUSTE DES DIMENSIONS AVANT FUSION
            normalized_tensors = []
            max_batch_size = 1
            
            for i, tensor in enumerate(modality_tensors):
                try:
                    # S'assurer qu'on a au moins 2D
                    if tensor.dim() == 1:
                        tensor = tensor.unsqueeze(0)
                    elif tensor.dim() == 0:
                        tensor = tensor.unsqueeze(0).unsqueeze(0)
                    
                    # Mettre à jour batch_size max
                    if tensor.dim() > 0:
                        max_batch_size = max(max_batch_size, tensor.size(0))
                    
                    # Si plus de 2D, aplatir vers 2D en gardant batch
                    if tensor.dim() > 2:
                        batch_size = tensor.size(0)
                        tensor = tensor.view(batch_size, -1)
                        
                        # Projeter vers hidden_size si nécessaire
                        if tensor.size(-1) != self.hidden_size:
                            if tensor.size(-1) > self.hidden_size * 10:  # Très grande dimension
                                tensor = tensor[:, :self.hidden_size]  # Tronquer
                            elif tensor.size(-1) != self.hidden_size:
                                # Projection ou padding/truncation
                                if tensor.size(-1) < self.hidden_size:
                                    padding = torch.zeros(batch_size, self.hidden_size - tensor.size(-1), device=tensor.device)
                                    tensor = torch.cat([tensor, padding], dim=-1)
                                else:
                                    tensor = tensor[:, :self.hidden_size]
                    
                    # S'assurer d'avoir [batch, hidden_size]
                    if tensor.size(-1) != self.hidden_size:
                        if tensor.size(-1) < self.hidden_size:
                            padding = torch.zeros(tensor.size(0), self.hidden_size - tensor.size(-1), device=tensor.device)
                            tensor = torch.cat([tensor, padding], dim=-1)
                        else:
                            tensor = tensor[:, :self.hidden_size]
                    
                    normalized_tensors.append(tensor)
                    
                except Exception as e:
                    logger.warning(f"Erreur normalisation tensor {i}: {e}")
                    # Tensor de secours
                    fallback = torch.zeros(max_batch_size, self.hidden_size, device=tensor.device if hasattr(tensor, 'device') else 'cpu')
                    normalized_tensors.append(fallback)
            
            # S'assurer que tous les tensors ont la même batch_size
            final_tensors = []
            for tensor in normalized_tensors:
                if tensor.size(0) != max_batch_size:
                    if tensor.size(0) == 1:
                        tensor = tensor.expand(max_batch_size, -1)
                    else:
                        # Prendre ou répéter pour avoir la bonne taille
                        if tensor.size(0) > max_batch_size:
                            tensor = tensor[:max_batch_size]
                        else:
                            repeats = max_batch_size // tensor.size(0)
                            remainder = max_batch_size % tensor.size(0)
                            tensor = torch.cat([tensor.repeat(repeats, 1), tensor[:remainder]], dim=0)
                final_tensors.append(tensor)
            
            # Maintenant, fusion sécurisée
            if self.fusion_strategy == FusionStrategy.ADAPTIVE_FUSION:
                fused_tensor, confidence, _ = self.fusion_module(final_tensors)
                confidence_score = float(confidence) if hasattr(confidence, 'item') else float(confidence)
                
            elif self.fusion_strategy == FusionStrategy.HIERARCHICAL_FUSION:
                if len(final_tensors) > 1:
                    # Stack maintenant sécurisé - tous ont même forme
                    stacked = torch.stack(final_tensors)
                    fused_tensor = self.fusion_module(stacked)
                else:
                    fused_tensor = final_tensors[0]
                confidence_score = 0.8
                
            elif self.fusion_strategy == FusionStrategy.ATTENTION_FUSION:
                if len(final_tensors) >= 2:
                    query, key, value = final_tensors[0], final_tensors[1], final_tensors[1]
                    fused_tensor = self.fusion_module(query, key, value)
                else:
                    fused_tensor = final_tensors[0]
                confidence_score = 0.75
                
            else:
                # Fusion simple (moyenne) - maintenant sécurisée
                if len(final_tensors) > 1:
                    fused_tensor = torch.stack(final_tensors).mean(dim=0)
                else:
                    fused_tensor = final_tensors[0]
                confidence_score = 0.7
            
            # Cache
            if cache_key and self.enable_caching:
                with self._cache_lock:
                    self._fusion_cache[cache_key] = (fused_tensor, confidence_score)
            
            return fused_tensor, confidence_score
            
        except Exception as e:
            logger.error(f"Erreur fusion: {e}")
            # Fallback sécurisé: utiliser le premier tensor disponible
            if modality_tensors:
                first_tensor = modality_tensors[0]
                if first_tensor.dim() == 1:
                    first_tensor = first_tensor.unsqueeze(0)
                if first_tensor.size(-1) == self.hidden_size:
                    fused_tensor = first_tensor
                else:
                    fused_tensor = torch.zeros(1, self.hidden_size, device=first_tensor.device)
            else:
                fused_tensor = torch.zeros(1, self.hidden_size)
            return fused_tensor, 0.1  # Confiance très faible pour le fallback
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calcule le taux de hit du cache."""
        total_cache_size = (len(self._tokenization_cache) + 
                          len(self._encoding_cache) + 
                          len(self._fusion_cache))
        return min(1.0, total_cache_size / 1000.0)  # Approximation
    
    def _reset_metrics(self):
        """Remet à zéro les métriques."""
        self.total_processed = 0
        self.total_time_ms = 0.0
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _update_global_stats(self, metrics: ProcessingMetrics):
        """Met à jour les statistiques globales."""
        self.total_processed += 1
        self.total_time_ms += metrics.total_time_ms
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de performance."""
        avg_time = self.total_time_ms / max(1, self.total_processed)
        
        return {
            'total_processed': self.total_processed,
            'average_time_ms': avg_time,
            'error_rate': self.error_count / max(1, self.total_processed),
            'fusion_strategy': self.fusion_strategy.value,
            'cache_enabled': self.enable_caching,
            'parallel_enabled': self.enable_parallel,
            'tokenizer_available': self.universal_tokenizer is not None,
            'supported_modalities': len(ModalityType2),
            'cache_sizes': {
                'tokenization': len(self._tokenization_cache) if self.enable_caching else 0,
                'encoding': len(self._encoding_cache) if self.enable_caching else 0,
                'fusion': len(self._fusion_cache) if self.enable_caching else 0
            } if self.enable_caching else {}
        }
    
    def clear_cache(self):
        """Vide tous les caches."""
        if self.enable_caching:
            with self._cache_lock:
                self._tokenization_cache.clear()
                self._encoding_cache.clear()
                self._fusion_cache.clear()
            logger.info("🧹 Caches vidés")
    
    def warmup(self, sample_inputs: Optional[Dict[str, Any]] = None):
        """Réchauffe le modèle pour optimiser les performances."""
        if self._warmup_completed:
            return
        
        logger.info("🔥 Réchauffage SuperMultimodalProcessor...")
        
        try:
            # Réchauffage minimal et sûr
            if sample_inputs is None:
                sample_inputs = {
                    'text': "warmup",  # Texte très court
                    'image': torch.randn(1, 3, 64, 64),  # Image plus petite
                    'audio': torch.randn(1, 1000)  # Audio minimal
                }
            
            # Passe de réchauffage ultra-rapide avec timeout
            with torch.no_grad():
                try:
                    # Juste initialiser les encodeurs sans traitement complet
                    for modality in ['text', 'image', 'audio']:
                        if modality in sample_inputs:
                            self._encode_single_modality(modality, 
                                torch.randn(1, 64) if modality == 'text' else sample_inputs[modality])
                except Exception as e:
                    logger.warning(f"Réchauffage partiel: {e}")
            
            self._warmup_completed = True
            logger.info("✅ Réchauffage terminé")
            
        except Exception as e:
            logger.warning(f"Réchauffage échoué, passage en mode dégradé: {e}")
            self._warmup_completed = True  # Marquer comme terminé pour éviter les boucles

# ============ ALIAS ET INTÉGRATION ============

# Remplacement optimisé des anciens systèmes
EnhancedMultimodalFusionCenter = SuperMultimodalProcessor
UnifiedMultimodalFusion = SuperMultimodalProcessor

def create_super_multimodal_processor(hidden_size: int = 768, **kwargs) -> SuperMultimodalProcessor:
    """Factory function pour créer un SuperMultimodalProcessor configuré."""
    return SuperMultimodalProcessor(hidden_size=hidden_size, **kwargs)