"""
NeuroLite Tokenization Metrics
==============================

Système de métriques pour les performances de tokenization.
"""

import time
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict, deque
import threading

class TokenizationMetrics:
    """Collecteur de métriques pour tokenization."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.lock = threading.Lock()
        
    def record_tokenization(self, result) -> None:
        """Enregistre une tokenization."""
        with self.lock:
            self.metrics['tokenization_times'].append(result.tokenization_time_ms)
            self.metrics['sequence_lengths'].append(result.sequence_length)
            self.metrics['compression_ratios'].append(result.compression_ratio)
            self.metrics['modalities'].append(result.modality.value)
            
    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des métriques."""
        with self.lock:
            if not self.metrics['tokenization_times']:
                return {}
            
            times = list(self.metrics['tokenization_times'])
            lengths = list(self.metrics['sequence_lengths'])
            
            return {
                'avg_tokenization_time_ms': sum(times) / len(times),
                'avg_sequence_length': sum(lengths) / len(lengths),
                'total_tokenizations': len(times),
                'modality_distribution': dict(
                    zip(*np.unique(list(self.metrics['modalities']), return_counts=True))
                ) if self.metrics['modalities'] else {}
            }