"""
Fonctions pour calculer les métriques de performance.
"""

import math
from typing import Dict

def compute_generative_metrics(eval_loss: float) -> Dict[str, float]:
    """
    Calcule les métriques pour les tâches de génération de langage.
    Pour l'instant, se concentre sur la perte et la perplexité.

    Args:
        eval_loss (float): La perte moyenne sur l'ensemble d'évaluation.

    Returns:
        Dict[str, float]: Un dictionnaire contenant la perte et la perplexité.
    """
    metrics = {"eval_loss": eval_loss}
    try:
        # La perplexité est l'exponentielle de la perte de cross-entropie
        perplexity = math.exp(eval_loss)
        metrics["perplexity"] = perplexity
    except (OverflowError, ValueError):
        # Si la perte est trop grande ou invalide, l'exponentielle peut échouer
        metrics["perplexity"] = float("inf")
        
    return metrics 