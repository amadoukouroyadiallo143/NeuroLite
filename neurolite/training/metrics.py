"""
NeuroLite AGI Metrics v2.0
Métriques spécialisées pour évaluer les performances AGI.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ..core.agi_model import AGIResponse

logger = logging.getLogger(__name__)

@dataclass
class AGIMetricsResult:
    """Résultat des métriques AGI."""
    # Métriques générales
    accuracy: float
    perplexity: float
    coherence_score: float
    
    # Métriques spécialisées AGI
    consciousness_coherence: float
    memory_precision: float
    memory_recall: float
    reasoning_validity: float
    multimodal_alignment: float
    response_confidence: float
    
    # Métriques temporelles
    processing_speed: float  # tokens/sec
    inference_time_ms: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convertit en dictionnaire pour logging."""
        return {
            "accuracy": self.accuracy,
            "perplexity": self.perplexity, 
            "coherence_score": self.coherence_score,
            "consciousness_coherence": self.consciousness_coherence,
            "memory_precision": self.memory_precision,
            "memory_recall": self.memory_recall,
            "reasoning_validity": self.reasoning_validity,
            "multimodal_alignment": self.multimodal_alignment,
            "response_confidence": self.response_confidence,
            "processing_speed": self.processing_speed,
            "inference_time_ms": self.inference_time_ms
        }

class AGIMetrics:
    """
    Système de métriques comprehensive pour NeuroLite AGI.
    Évalue tous les aspects: génération, conscience, mémoire, raisonnement.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        
        # Buffers pour accumulation
        self.reset_buffers()
        
        logger.info("📊 AGIMetrics initialisé")
    
    def reset_buffers(self):
        """Remet à zéro les buffers d'accumulation."""
        self.predictions_buffer = []
        self.targets_buffer = []
        self.consciousness_buffer = []
        self.memory_buffer = []
        self.reasoning_buffer = []
        self.processing_times_buffer = []
        
    def compute_training_metrics(
        self, 
        response: AGIResponse, 
        targets: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calcule les métriques d'entraînement."""
        
        metrics = {}
        
        # 1. Métriques de base
        if "primary_target" in targets:
            base_metrics = self._compute_base_metrics(
                response.primary_output,
                targets["primary_target"]
            )
            metrics.update(base_metrics)
        
        # 2. Métriques de conscience
        if hasattr(response, 'consciousness_level'):
            consciousness_metrics = self._compute_consciousness_metrics(
                response.consciousness_level,
                targets.get("consciousness_target", 0.5)
            )
            metrics.update(consciousness_metrics)
        
        # 3. Métriques de mémoire
        if response.memory_insights:
            memory_metrics = self._compute_memory_metrics(
                response.memory_insights,
                targets.get("memory_targets", {})
            )
            metrics.update(memory_metrics)
        
        # 4. Métriques de raisonnement
        if response.reasoning_chain:
            reasoning_metrics = self._compute_reasoning_metrics(
                response.reasoning_chain,
                targets.get("reasoning_targets", [])
            )
            metrics.update(reasoning_metrics)
        
        # 5. Métriques de performance
        if hasattr(response, 'processing_time_ms'):
            perf_metrics = self._compute_performance_metrics(response)
            metrics.update(perf_metrics)
        
        return metrics
    
    def compute_validation_metrics(
        self,
        response: AGIResponse,
        targets: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calcule les métriques de validation (plus détaillées)."""
        
        # Base des métriques d'entraînement
        metrics = self.compute_training_metrics(response, targets)
        
        # Métriques additionnelles pour validation
        if "primary_target" in targets:
            # Diversité des réponses
            diversity_score = self._compute_diversity_score(
                response.primary_output,
                response.alternative_responses
            )
            metrics["diversity_score"] = diversity_score
            
            # Stabilité (consistance entre runs)
            if len(self.predictions_buffer) > 0:
                stability_score = self._compute_stability_score(
                    response.primary_output,
                    self.predictions_buffer[-5:]  # 5 dernières prédictions
                )
                metrics["stability_score"] = stability_score
        
        # Accumulation pour métriques d'époque
        self.predictions_buffer.append(response.primary_output)
        if "primary_target" in targets:
            self.targets_buffer.append(targets["primary_target"])
        
        return metrics
    
    def _compute_base_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Calcule les métriques de base."""
        
        metrics = {}
        
        # Accuracy (approximation pour données continues)
        if predictions.shape == targets.shape:
            # MSE et MAE
            mse = F.mse_loss(predictions, targets).item()
            mae = F.l1_loss(predictions, targets).item()
            
            metrics["mse"] = mse
            metrics["mae"] = mae
            
            # Pseudo-accuracy (basée sur seuil)
            threshold = 0.1  # 10% de tolérance
            correct = (torch.abs(predictions - targets) < threshold).float()
            accuracy = correct.mean().item()
            metrics["accuracy"] = accuracy
            
            # Corrélation
            if predictions.numel() > 1:
                correlation = torch.corrcoef(torch.stack([
                    predictions.flatten(),
                    targets.flatten()
                ]))[0, 1].item()
                if not torch.isnan(torch.tensor(correlation)):
                    metrics["correlation"] = correlation
        
        # Perplexité (approximation)
        if predictions.dim() > 1 and predictions.size(-1) > 1:
            # Convertir en probabilités via softmax
            probs = F.softmax(predictions, dim=-1)
            # Éviter log(0) 
            probs = torch.clamp(probs, min=1e-8)
            entropy = -torch.sum(probs * torch.log(probs), dim=-1)
            perplexity = torch.exp(entropy.mean()).item()
            metrics["perplexity"] = perplexity
        
        return metrics
    
    def _compute_consciousness_metrics(
        self,
        consciousness_level: float,
        target_level: float = 0.5
    ) -> Dict[str, float]:
        """Calcule les métriques de conscience."""
        
        metrics = {}
        
        # Cohérence de conscience (proximity to target)
        if isinstance(consciousness_level, (int, float)):
            coherence = 1.0 - abs(consciousness_level - target_level)
            metrics["consciousness_coherence"] = max(0.0, coherence)
            
            # Stabilité (évite les valeurs extrêmes)
            stability = 1.0 - abs(consciousness_level - 0.5)
            metrics["consciousness_stability"] = stability
            
            # Activité (évite les valeurs trop faibles)
            activity = min(1.0, consciousness_level * 2.0)
            metrics["consciousness_activity"] = activity
        
        return metrics
    
    def _compute_memory_metrics(
        self,
        memory_insights: Dict[str, Any],
        memory_targets: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calcule les métriques de mémoire."""
        
        metrics = {}
        
        # Précision de rappel
        if "recalled_items" in memory_insights:
            recalled_count = len(memory_insights["recalled_items"])
            if "expected_count" in memory_targets:
                expected_count = memory_targets["expected_count"]
                precision = min(1.0, recalled_count / max(1, expected_count))
                metrics["memory_precision"] = precision
            else:
                # Score basé sur le nombre d'éléments rappelés
                metrics["memory_precision"] = min(1.0, recalled_count / 10.0)
        
        # Pertinence des souvenirs
        if "relevance_scores" in memory_insights:
            relevance_scores = memory_insights["relevance_scores"]
            if isinstance(relevance_scores, (list, tuple)) and len(relevance_scores) > 0:
                avg_relevance = sum(relevance_scores) / len(relevance_scores)
                metrics["memory_relevance"] = avg_relevance
        
        # Diversité des souvenirs
        if "memory_types" in memory_insights:
            memory_types = memory_insights["memory_types"]
            if isinstance(memory_types, (list, tuple)):
                unique_types = len(set(memory_types))
                diversity = unique_types / max(1, len(memory_types))
                metrics["memory_diversity"] = diversity
        
        # Score de rappel global
        recall_score = 0.0
        if "total_memories" in memory_insights and "accessed_memories" in memory_insights:
            total = memory_insights["total_memories"]
            accessed = memory_insights["accessed_memories"]
            if total > 0:
                recall_score = accessed / total
        metrics["memory_recall"] = recall_score
        
        return metrics
    
    def _compute_reasoning_metrics(
        self,
        reasoning_chain: List[str],
        reasoning_targets: List[str] = []
    ) -> Dict[str, float]:
        """Calcule les métriques de raisonnement."""
        
        metrics = {}
        
        # Validité de la chaîne de raisonnement
        chain_length = len(reasoning_chain)
        
        if chain_length > 0:
            # Score de longueur (ni trop court ni trop long)
            optimal_length = 5
            length_score = 1.0 - abs(chain_length - optimal_length) / optimal_length
            metrics["reasoning_length_score"] = max(0.0, length_score)
            
            # Cohérence (pas de répétitions excessives)
            unique_steps = len(set(reasoning_chain))
            coherence = unique_steps / chain_length
            metrics["reasoning_coherence"] = coherence
            
            # Complexité (diversité des concepts)
            complexity_score = min(1.0, unique_steps / 3.0)
            metrics["reasoning_complexity"] = complexity_score
            
            # Score global de validité
            validity = (length_score + coherence + complexity_score) / 3.0
            metrics["reasoning_validity"] = validity
        else:
            # Pas de raisonnement
            metrics["reasoning_validity"] = 0.0
            metrics["reasoning_coherence"] = 0.0
            metrics["reasoning_complexity"] = 0.0
        
        return metrics
    
    def _compute_performance_metrics(
        self,
        response: AGIResponse
    ) -> Dict[str, float]:
        """Calcule les métriques de performance."""
        
        metrics = {}
        
        # Temps de traitement
        if hasattr(response, 'processing_time_ms'):
            processing_time = response.processing_time_ms
            metrics["processing_time_ms"] = processing_time
            
            # Score d'efficacité (inverse du temps)
            efficiency = 1000.0 / max(1.0, processing_time)  # Plus c'est rapide, mieux c'est
            metrics["processing_efficiency"] = min(1.0, efficiency)
        
        # Confiance de réponse
        if hasattr(response, 'confidence_score'):
            metrics["response_confidence"] = response.confidence_score
        
        return metrics
    
    def _compute_diversity_score(
        self,
        primary_output: torch.Tensor,
        alternative_outputs: List[torch.Tensor]
    ) -> float:
        """Calcule la diversité des réponses."""
        
        if not alternative_outputs:
            return 0.0
        
        diversities = []
        
        for alt_output in alternative_outputs:
            if alt_output.shape == primary_output.shape:
                # Distance cosine entre sortie principale et alternative
                similarity = F.cosine_similarity(
                    primary_output.flatten(),
                    alt_output.flatten(),
                    dim=0
                ).item()
                diversity = 1.0 - abs(similarity)  # Plus différent = plus diverse
                diversities.append(diversity)
        
        return sum(diversities) / len(diversities) if diversities else 0.0
    
    def _compute_stability_score(
        self,
        current_output: torch.Tensor,
        previous_outputs: List[torch.Tensor]
    ) -> float:
        """Calcule la stabilité des réponses."""
        
        if not previous_outputs:
            return 1.0  # Parfaitement stable s'il n'y a qu'une réponse
        
        similarities = []
        
        for prev_output in previous_outputs:
            if prev_output.shape == current_output.shape:
                similarity = F.cosine_similarity(
                    current_output.flatten(),
                    prev_output.flatten(),
                    dim=0
                ).item()
                similarities.append(abs(similarity))
        
        # Stabilité = similarité moyenne avec les réponses précédentes
        return sum(similarities) / len(similarities) if similarities else 1.0
    
    def compute_epoch_metrics(self) -> Dict[str, float]:
        """Calcule les métriques agrégées sur l'époque."""
        
        metrics = {}
        
        # Métriques générales si on a des prédictions et targets
        if len(self.predictions_buffer) > 0 and len(self.targets_buffer) > 0:
            # Stack tous les tensors
            try:
                all_predictions = torch.stack(self.predictions_buffer)
                all_targets = torch.stack(self.targets_buffer)
                
                # MSE global
                epoch_mse = F.mse_loss(all_predictions, all_targets).item()
                metrics["epoch_mse"] = epoch_mse
                
                # Corrélation globale
                if all_predictions.numel() > 1:
                    correlation = torch.corrcoef(torch.stack([
                        all_predictions.flatten(),
                        all_targets.flatten()
                    ]))[0, 1].item()
                    if not torch.isnan(torch.tensor(correlation)):
                        metrics["epoch_correlation"] = correlation
                        
            except Exception as e:
                logger.warning(f"Erreur calcul métriques d'époque: {e}")
        
        # Reset buffers après calcul
        self.reset_buffers()
        
        return metrics
    
    def compute_comprehensive_report(
        self,
        responses: List[AGIResponse],
        targets_list: List[Dict[str, Any]]
    ) -> AGIMetricsResult:
        """Calcule un rapport complet de métriques."""
        
        if not responses:
            # Valeurs par défaut si pas de données
            return AGIMetricsResult(
                accuracy=0.0, perplexity=float('inf'), coherence_score=0.0,
                consciousness_coherence=0.0, memory_precision=0.0, memory_recall=0.0,
                reasoning_validity=0.0, multimodal_alignment=0.0, response_confidence=0.0,
                processing_speed=0.0, inference_time_ms=0.0
            )
        
        # Calcul des métriques pour chaque réponse
        all_metrics = []
        for i, response in enumerate(responses):
            targets = targets_list[i] if i < len(targets_list) else {}
            metrics = self.compute_validation_metrics(response, targets)
            all_metrics.append(metrics)
        
        # Agrégation
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            if values:
                aggregated[key] = sum(values) / len(values)
        
        # Construction du résultat
        return AGIMetricsResult(
            accuracy=aggregated.get("accuracy", 0.0),
            perplexity=aggregated.get("perplexity", float('inf')),
            coherence_score=aggregated.get("reasoning_coherence", 0.0),
            consciousness_coherence=aggregated.get("consciousness_coherence", 0.0),
            memory_precision=aggregated.get("memory_precision", 0.0),
            memory_recall=aggregated.get("memory_recall", 0.0),
            reasoning_validity=aggregated.get("reasoning_validity", 0.0),
            multimodal_alignment=aggregated.get("correlation", 0.0),
            response_confidence=aggregated.get("response_confidence", 0.0),
            processing_speed=1000.0 / aggregated.get("processing_time_ms", 1000.0),
            inference_time_ms=aggregated.get("processing_time_ms", 0.0)
        )