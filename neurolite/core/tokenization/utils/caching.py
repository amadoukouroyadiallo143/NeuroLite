"""
NeuroLite Embedding Cache System
===============================

Système de cache intelligent pour les embeddings.
"""

import time
import threading
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """Cache intelligent pour embeddings avec gestion mémoire."""
    
    def __init__(self, max_size_mb: float = 512.0):
        self.max_size_mb = max_size_mb
        self.cache = {}
        self.access_times = {}
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()
        
    def get(self, key: str) -> Optional[Any]:
        """Récupère un élément du cache."""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                self.hits += 1
                return self.cache[key]
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any) -> bool:
        """Ajoute un élément au cache."""
        with self.lock:
            self.cache[key] = value
            self.access_times[key] = time.time()
            self._cleanup_if_needed()
            return True
    
    def _cleanup_if_needed(self):
        """Nettoie le cache si nécessaire."""
        # Implémentation simplifiée
        if len(self.cache) > 1000:  # Limite arbitraire
            # Supprimer les plus anciens
            oldest_keys = sorted(self.access_times.keys(), 
                               key=lambda k: self.access_times[k])[:100]
            for key in oldest_keys:
                del self.cache[key]
                del self.access_times[key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / max(1, total)
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'cache_size': len(self.cache),
                'max_size_mb': self.max_size_mb
            }
    
    def __contains__(self, key: str) -> bool:
        return key in self.cache