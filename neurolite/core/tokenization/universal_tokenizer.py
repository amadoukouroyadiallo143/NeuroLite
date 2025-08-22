"""
NeuroLite Universal Tokenizer
============================

Tokenizer universel qui orchestre tous les tokenizers spécialisés.
Point d'entrée principal pour la tokenization de tous types de données.
"""

import time
import logging
import torch
from typing import Any, Dict, List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .base_tokenizer import BaseTokenizer, TokenizerConfig, TokenizationResult, ModalityType
from .tokenizer_registry import TokenizerRegistry, get_global_registry
from .modality_detectors import ModalityDetector, detect_modality
from .utils.caching import EmbeddingCache
from .utils.metrics import TokenizationMetrics

logger = logging.getLogger(__name__)

class UniversalTokenizer:
    """
    Tokenizer universel qui gère tous types de données automatiquement.
    Utilise la détection de modalité et le registre pour choisir le bon tokenizer.
    """
    
    def __init__(self, 
                 config: Optional[TokenizerConfig] = None,
                 registry: Optional[TokenizerRegistry] = None,
                 detector: Optional[ModalityDetector] = None):
        """
        Initialise le tokenizer universel.
        
        Args:
            config: Configuration globale
            registry: Registre des tokenizers (défaut: global)
            detector: Détecteur de modalités (défaut: nouveau)
        """
        self.config = config or TokenizerConfig()
        self.registry = registry or get_global_registry()
        self.detector = detector or ModalityDetector()
        
        # Cache et métriques
        self.cache = EmbeddingCache(max_size_mb=self.config.cache_size_mb) if self.config.cache_embeddings else None
        self.metrics = TokenizationMetrics() if self.config.enable_metrics else None
        
        # Pool de threads pour traitement parallèle
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.num_workers) if self.config.enable_parallel else None
        
        # Statistiques
        self.total_tokenizations = 0
        self.successful_tokenizations = 0
        self.failed_tokenizations = 0
        self.tokenizations_by_modality = {}
        
        # Verrous pour thread-safety
        self._stats_lock = threading.Lock()
        
        logger.info(f"UniversalTokenizer initialisé avec {len(self.registry.list_tokenizers())} tokenizers")
    
    def tokenize(self, 
                 data: Any,
                 modality: Optional[ModalityType] = None,
                 strategy: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None,
                 **kwargs) -> TokenizationResult:
        """
        Tokenise des données de n'importe quel type.
        
        Args:
            data: Données à tokeniser
            modality: Modalité forcée (optionnel, sinon auto-détectée)
            strategy: Stratégie de tokenization préférée
            context: Contexte additionnel pour la détection
            **kwargs: Arguments spécifiques au tokenizer
            
        Returns:
            TokenizationResult: Résultat complet de la tokenization
        """
        start_time = time.time()
        
        try:
            with self._stats_lock:
                self.total_tokenizations += 1
            
            # 1. Détection de modalité si non fournie
            if modality is None:
                detected_modality, confidence = self.detector.detect(data, context)
                if confidence < 0.5:
                    logger.warning(f"Détection de modalité incertaine: {detected_modality.value} (confiance: {confidence:.3f})")
                modality = detected_modality
            
            # 2. Récupération du tokenizer approprié
            tokenizer = self.registry.get_tokenizer_for_modality(modality, self.config, strategy)
            if tokenizer is None:
                raise ValueError(f"Aucun tokenizer disponible pour modalité {modality.value}")
            
            # 3. Vérification du cache
            cache_key = self._generate_cache_key(data, modality, strategy, kwargs)
            if self.cache and cache_key in self.cache:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    logger.debug(f"Résultat récupéré du cache pour {modality.value}")
                    return cached_result
            
            # 4. Tokenization
            result = tokenizer.tokenize(data, **kwargs)
            
            # 5. Post-traitement et enrichissement
            result = self._enrich_result(result, modality, strategy, start_time, tokenizer)
            
            # 6. Mise en cache
            if self.cache and cache_key:
                self.cache.put(cache_key, result)
            
            # 7. Métriques
            if self.metrics:
                self.metrics.record_tokenization(result)
            
            # 8. Statistiques
            with self._stats_lock:
                self.successful_tokenizations += 1
                self.tokenizations_by_modality[modality.value] = self.tokenizations_by_modality.get(modality.value, 0) + 1
            
            logger.debug(f"Tokenization réussie: {modality.value} -> {result.sequence_length} tokens")
            return result
            
        except Exception as e:
            with self._stats_lock:
                self.failed_tokenizations += 1
            
            error_msg = f"Erreur tokenization: {e}"
            logger.error(error_msg)
            
            # Retourner un résultat d'erreur
            return TokenizationResult(
                tokens=[],
                modality=modality or ModalityType.UNKNOWN,
                tokenization_time_ms=(time.time() - start_time) * 1000,
                metadata={'error': str(e)}
            )
    
    def detokenize(self, 
                   result: TokenizationResult,
                   **kwargs) -> Any:
        """
        Reconstruit les données originales à partir d'un résultat de tokenization.
        
        Args:
            result: Résultat de tokenization
            **kwargs: Arguments additionnels
            
        Returns:
            Données reconstruites
        """
        try:
            # Récupérer le tokenizer approprié
            tokenizer = self.registry.get_tokenizer_for_modality(result.modality, self.config, 
                                                               result.strategy.value if result.strategy else None)
            if tokenizer is None:
                raise ValueError(f"Aucun tokenizer pour détokenization {result.modality.value}")
            
            # Détokenizer
            return tokenizer.detokenize(result.tokens, **kwargs)
            
        except Exception as e:
            logger.error(f"Erreur détokenization: {e}")
            return None
    
    def tokenize_batch(self, 
                       data_list: List[Any],
                       modalities: Optional[List[ModalityType]] = None,
                       strategies: Optional[List[str]] = None,
                       contexts: Optional[List[Dict[str, Any]]] = None,
                       **kwargs) -> List[TokenizationResult]:
        """
        Tokenise un batch de données en parallèle.
        
        Args:
            data_list: Liste de données à tokeniser
            modalities: Modalités forcées (optionnel)
            strategies: Stratégies préférées (optionnel)
            contexts: Contextes pour détection (optionnel)
            **kwargs: Arguments communs
            
        Returns:
            Liste des résultats de tokenization
        """
        if not self.thread_pool:
            # Mode séquentiel
            results = []
            for i, data in enumerate(data_list):
                modality = modalities[i] if modalities else None
                strategy = strategies[i] if strategies else None
                context = contexts[i] if contexts else None
                result = self.tokenize(data, modality, strategy, context, **kwargs)
                results.append(result)
            return results
        
        # Mode parallèle
        futures = []
        
        for i, data in enumerate(data_list):
            modality = modalities[i] if modalities else None
            strategy = strategies[i] if strategies else None
            context = contexts[i] if contexts else None
            
            future = self.thread_pool.submit(
                self.tokenize, data, modality, strategy, context, **kwargs
            )
            futures.append(future)
        
        # Récupération des résultats
        results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=30)  # Timeout de sécurité
                results.append(result)
            except Exception as e:
                logger.error(f"Erreur tokenization parallèle: {e}")
                results.append(TokenizationResult(
                    tokens=[],
                    metadata={'error': str(e)}
                ))
        
        return results
    
    def get_embeddings(self, 
                      data: Any,
                      modality: Optional[ModalityType] = None,
                      **kwargs) -> Optional[torch.Tensor]:
        """
        Récupère directement les embeddings pour des données.
        
        Args:
            data: Données à encoder
            modality: Modalité forcée
            **kwargs: Arguments additionnels
            
        Returns:
            torch.Tensor: Embeddings ou None en cas d'erreur
        """
        try:
            # Tokeniser d'abord
            result = self.tokenize(data, modality, **kwargs)
            
            # Récupérer le tokenizer pour les embeddings
            tokenizer = self.registry.get_tokenizer_for_modality(result.modality, self.config)
            if tokenizer is None:
                return None
            
            # Générer les embeddings
            embeddings = tokenizer.get_embeddings(result.tokens)
            
            # Mettre à jour le résultat avec les embeddings
            result.embeddings = embeddings
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Erreur génération embeddings: {e}")
            return None
    
    def estimate_tokens(self, 
                       data: Any,
                       modality: Optional[ModalityType] = None) -> int:
        """
        Estime le nombre de tokens sans tokenizer complètement.
        
        Args:
            data: Données à analyser
            modality: Modalité forcée
            
        Returns:
            int: Estimation du nombre de tokens
        """
        try:
            # Détection rapide de modalité
            if modality is None:
                modality, _ = self.detector.detect(data)
            
            # Estimations heuristiques par modalité
            if modality == ModalityType.TEXT:
                if isinstance(data, str):
                    # Approximation: ~0.75 tokens par mot
                    return int(len(data.split()) * 0.75)
                    
            elif modality == ModalityType.IMAGE:
                if hasattr(data, 'shape'):
                    h, w = data.shape[-2:]
                    # Approximation: patches 16x16
                    return (h // 16) * (w // 16)
                    
            elif modality == ModalityType.AUDIO:
                if hasattr(data, 'shape'):
                    # Approximation: 1 token par 320 échantillons
                    return data.shape[-1] // 320
            
            return 0
            
        except Exception:
            return 0
    
    def get_supported_modalities(self) -> List[ModalityType]:
        """Retourne la liste des modalités supportées."""
        modalities = set()
        for tokenizer_info in self.registry.list_tokenizers():
            modalities.add(ModalityType(tokenizer_info['modality']))
        return list(modalities)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques complètes du tokenizer universel."""
        with self._stats_lock:
            base_stats = {
                'total_tokenizations': self.total_tokenizations,
                'successful_tokenizations': self.successful_tokenizations,
                'failed_tokenizations': self.failed_tokenizations,
                'success_rate': self.successful_tokenizations / max(1, self.total_tokenizations),
                'tokenizations_by_modality': self.tokenizations_by_modality,
                'supported_modalities': [m.value for m in self.get_supported_modalities()]
            }
        
        # Ajouter statistiques des composants
        if self.cache:
            base_stats['cache_stats'] = self.cache.get_statistics()
        
        if self.metrics:
            base_stats['tokenization_metrics'] = self.metrics.get_summary()
        
        base_stats['registry_stats'] = self.registry.get_statistics()
        base_stats['detector_stats'] = self.detector.get_detection_statistics()
        
        return base_stats
    
    def _generate_cache_key(self, 
                           data: Any,
                           modality: ModalityType,
                           strategy: Optional[str],
                           kwargs: Dict[str, Any]) -> Optional[str]:
        """Génère une clé de cache pour les données."""
        try:
            import hashlib
            
            # Créer une représentation hashable des données
            if isinstance(data, str):
                data_repr = data.encode()
            elif hasattr(data, 'tobytes'):
                data_repr = data.tobytes()
            elif hasattr(data, '__str__'):
                data_repr = str(data).encode()
            else:
                return None
            
            # Créer le hash
            hasher = hashlib.md5()
            hasher.update(data_repr)
            hasher.update(modality.value.encode())
            if strategy:
                hasher.update(strategy.encode())
            hasher.update(str(sorted(kwargs.items())).encode())
            
            return hasher.hexdigest()
            
        except Exception:
            return None
    
    def _enrich_result(self, 
                      result: TokenizationResult,
                      modality: ModalityType,
                      strategy: Optional[str],
                      start_time: float,
                      tokenizer: BaseTokenizer) -> TokenizationResult:
        """Enrichit le résultat avec des métadonnées additionnelles."""
        
        # Calculer le temps de traitement
        result.tokenization_time_ms = (time.time() - start_time) * 1000
        
        # Ajouter métadonnées de modalité et stratégie
        result.modality = modality
        if strategy:
            try:
                result.strategy = next(s for s in tokenizer.get_supported_strategies() if s.value == strategy)
            except StopIteration:
                pass
        
        # Calculer des métriques
        if hasattr(result, 'tokens') and result.tokens is not None:
            if hasattr(result.tokens, '__len__'):
                result.sequence_length = len(result.tokens)
            result.vocab_size = tokenizer.get_vocab_size()
        
        # Métadonnées additionnelles
        result.metadata.update({
            'tokenizer_class': tokenizer.__class__.__name__,
            'config_version': self.config.__dict__,
            'universal_tokenizer_version': '1.0.0'
        })
        
        return result
    
    def __del__(self):
        """Nettoyage lors de la destruction."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)


# ============ FONCTION FACTORY GLOBALE ============

_global_tokenizer = None

def get_universal_tokenizer(config: Optional[TokenizerConfig] = None) -> UniversalTokenizer:
    """
    Retourne une instance singleton du tokenizer universel.
    
    Args:
        config: Configuration optionnelle
        
    Returns:
        UniversalTokenizer: Instance configurée
    """
    global _global_tokenizer
    
    if _global_tokenizer is None:
        try:
            if config is None:
                # Configuration par défaut simple
                config = TokenizerConfig(
                    vocab_size=50000,
                    max_sequence_length=4096,
                    padding_strategy="max_length",
                    truncation=True,
                    return_tensors="torch",
                    cache_embeddings=True,
                    enable_metrics=True
                )
            
            _global_tokenizer = UniversalTokenizer(config=config)
            logger.info("✅ Universal Tokenizer global créé avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur création Universal Tokenizer: {e}")
            # Créer un tokenizer minimal fonctionnel
            _global_tokenizer = _create_minimal_tokenizer()
    
    return _global_tokenizer


def _create_minimal_tokenizer():
    """Crée un tokenizer minimal en cas d'erreur."""
    
    class MinimalTokenizer:
        """Tokenizer minimal de fallback."""
        
        def __init__(self):
            self.config = TokenizerConfig()
            
        def tokenize(self, data, modality=None, **kwargs):
            """Tokenization basique."""
            try:
                if isinstance(data, str):
                    # Tokenization simple du texte
                    tokens = data.split()[:self.config.max_sequence_length]
                    embeddings = torch.randn(len(tokens), 768)  # Embeddings factices
                else:
                    # Pour autres types, créer des embeddings factices
                    tokens = ["<unk>"]
                    embeddings = torch.randn(1, 768)
                
                return TokenizationResult(
                    tokens=tokens,
                    embeddings=embeddings,
                    modality=modality or ModalityType.TEXT,
                    metadata={"fallback": True}
                )
            except Exception:
                # Ultime fallback
                return TokenizationResult(
                    tokens=["<error>"],
                    embeddings=torch.zeros(1, 768),
                    modality=ModalityType.TEXT,
                    metadata={"error": True}
                )
    
    logger.warning("⚠️ Utilisation du tokenizer minimal de fallback")
    return MinimalTokenizer()