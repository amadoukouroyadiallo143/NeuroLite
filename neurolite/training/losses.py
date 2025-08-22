"""
NeuroLite AGI Losses v2.0
Fonctions de perte spécialisées pour l'entraînement AGI multimodal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import logging

from ..core.agi_model import AGIResponse

logger = logging.getLogger(__name__)

class AGIMultiModalLoss(nn.Module):
    """
    Fonction de perte composite pour NeuroLite AGI.
    Combine pertes multimodales, conscience, mémoire et raisonnement.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        temperature: float = 1.0,
        use_consciousness_loss: bool = True,
        use_memory_loss: bool = True,
        use_reasoning_loss: bool = True
    ):
        super().__init__()
        
        # Poids des différentes pertes
        self.weights = weights or {
            "generation": 1.0,      # Perte génération principale
            "consciousness": 0.3,   # Cohérence de conscience
            "memory": 0.2,          # Précision mémoire
            "reasoning": 0.4,       # Validité raisonnement
            "consistency": 0.1,     # Consistance multimodale
            "alignment": 0.2        # Alignement cross-modal
        }
        
        self.temperature = temperature
        self.use_consciousness_loss = use_consciousness_loss
        self.use_memory_loss = use_memory_loss 
        self.use_reasoning_loss = use_reasoning_loss
        
        # Pertes de base
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
        logger.info(f"🎯 AGIMultiModalLoss initialisé avec poids: {self.weights}")
    
    def forward(
        self, 
        agi_response: AGIResponse, 
        targets: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Calcule les pertes AGI composites.
        
        Args:
            agi_response: Réponse du modèle AGI
            targets: Targets d'entraînement
            
        Returns:
            Dict des pertes calculées
        """
        
        losses = {}
        total_loss = torch.tensor(0.0, device=agi_response.primary_output.device)
        
        # 1. Perte de génération principale
        generation_loss = self._compute_generation_loss(
            agi_response.primary_output, 
            targets.get("primary_target")
        )
        losses["generation_loss"] = generation_loss
        total_loss += self.weights["generation"] * generation_loss
        
        # 2. Perte de conscience (cohérence)
        if self.use_consciousness_loss and hasattr(agi_response, 'consciousness_level'):
            consciousness_loss = self._compute_consciousness_loss(
                agi_response.consciousness_level,
                targets.get("consciousness_target", 0.5)
            )
            losses["consciousness_loss"] = consciousness_loss
            total_loss += self.weights["consciousness"] * consciousness_loss
        
        # 3. Perte de mémoire (précision rappel)
        if self.use_memory_loss and agi_response.memory_insights:
            memory_loss = self._compute_memory_loss(
                agi_response.memory_insights,
                targets.get("memory_targets", {})
            )
            losses["memory_loss"] = memory_loss
            total_loss += self.weights["memory"] * memory_loss
        
        # 4. Perte de raisonnement (cohérence logique)
        if self.use_reasoning_loss and agi_response.reasoning_chain:
            reasoning_loss = self._compute_reasoning_loss(
                agi_response.reasoning_chain,
                targets.get("reasoning_target", [])
            )
            losses["reasoning_loss"] = reasoning_loss
            total_loss += self.weights["reasoning"] * reasoning_loss
        
        # 5. Consistance entre modalités
        if len(agi_response.alternative_responses) > 0:
            consistency_loss = self._compute_consistency_loss(
                agi_response.primary_output,
                agi_response.alternative_responses
            )
            losses["consistency_loss"] = consistency_loss
            total_loss += self.weights["consistency"] * consistency_loss
        
        # 6. Alignement cross-modal
        if "multimodal_targets" in targets:
            alignment_loss = self._compute_alignment_loss(
                agi_response.primary_output,
                targets["multimodal_targets"]
            )
            losses["alignment_loss"] = alignment_loss
            total_loss += self.weights["alignment"] * alignment_loss
        
        losses["total_loss"] = total_loss
        
        return losses
    
    def _compute_generation_loss(
        self, 
        output: torch.Tensor, 
        target: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Perte de génération principale."""
        if target is None:
            return torch.tensor(0.0, device=output.device)
        
        # Adapter selon les shapes
        if output.shape != target.shape:
            # Truncate ou pad selon besoin
            min_size = min(output.size(-1), target.size(-1))
            output = output[..., :min_size]
            target = target[..., :min_size]
        
        return self.mse_loss(output, target)
    
    def _compute_consciousness_loss(
        self,
        consciousness_level: float,
        target_level: float
    ) -> torch.Tensor:
        """Perte de cohérence de conscience."""
        
        # Conversion en tensors
        if isinstance(consciousness_level, (int, float)):
            consciousness_level = torch.tensor(consciousness_level, dtype=torch.float32)
        if isinstance(target_level, (int, float)):
            target_level = torch.tensor(target_level, dtype=torch.float32)
        
        # Loss de cohérence (proche du target souhaité)
        coherence_loss = F.mse_loss(consciousness_level, target_level)
        
        # Régularisation: éviter les extrêmes (0 ou 1)
        regularization = torch.abs(consciousness_level - 0.5) * 0.1
        
        return coherence_loss + regularization.mean()
    
    def _compute_memory_loss(
        self,
        memory_insights: Dict[str, Any],
        memory_targets: Dict[str, Any]
    ) -> torch.Tensor:
        """Perte de précision mémoire."""
        
        if not memory_targets:
            return torch.tensor(0.0)
        
        total_memory_loss = torch.tensor(0.0)
        
        # Perte de rappel (recall accuracy)
        if "recalled_items" in memory_insights and "expected_items" in memory_targets:
            recalled = memory_insights["recalled_items"]
            expected = memory_targets["expected_items"]
            
            # Similarité cosine entre éléments rappelés et attendus
            if len(recalled) > 0 and len(expected) > 0:
                recall_loss = 1.0 - F.cosine_similarity(
                    torch.stack(recalled).mean(0),
                    torch.stack(expected).mean(0),
                    dim=0
                )
                total_memory_loss += recall_loss
        
        # Perte de relevance (pertinence des souvenirs)
        if "relevance_scores" in memory_insights:
            relevance_scores = memory_insights["relevance_scores"]
            if isinstance(relevance_scores, list) and len(relevance_scores) > 0:
                # Les scores de relevance devraient être élevés
                relevance_loss = 1.0 - torch.tensor(relevance_scores).mean()
                total_memory_loss += relevance_loss
        
        return total_memory_loss
    
    def _compute_reasoning_loss(
        self,
        reasoning_chain: List[str],
        reasoning_targets: List[str]
    ) -> torch.Tensor:
        """Perte de cohérence de raisonnement."""
        
        if not reasoning_targets:
            return torch.tensor(0.0)
        
        # Pour l'instant, loss basée sur longueur et cohérence
        chain_length = len(reasoning_chain)
        target_length = len(reasoning_targets)
        
        # Perte de longueur (chaîne de raisonnement appropriée)
        length_loss = F.mse_loss(
            torch.tensor(chain_length, dtype=torch.float32),
            torch.tensor(target_length, dtype=torch.float32)
        )
        
        # Perte de cohérence: les étapes doivent être logiques
        coherence_loss = torch.tensor(0.0)
        if chain_length > 1:
            # Pénalité si trop de répétitions ou incohérence
            unique_steps = len(set(reasoning_chain))
            coherence_loss = max(0, (chain_length - unique_steps) / chain_length)
        
        return length_loss + coherence_loss
    
    def _compute_consistency_loss(
        self,
        primary_output: torch.Tensor,
        alternative_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """Perte de consistance entre les sorties."""
        
        if not alternative_outputs:
            return torch.tensor(0.0, device=primary_output.device)
        
        total_consistency_loss = torch.tensor(0.0, device=primary_output.device)
        
        for alt_output in alternative_outputs:
            # Similarité entre sortie principale et alternatives
            if alt_output.shape == primary_output.shape:
                consistency = F.cosine_similarity(
                    primary_output.flatten(),
                    alt_output.flatten(),
                    dim=0
                )
                # On veut de la diversité mais pas trop
                target_similarity = 0.7
                consistency_loss = F.mse_loss(consistency, torch.tensor(target_similarity))
                total_consistency_loss += consistency_loss
        
        return total_consistency_loss / len(alternative_outputs) if alternative_outputs else total_consistency_loss
    
    def _compute_alignment_loss(
        self,
        output: torch.Tensor,
        multimodal_targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Perte d'alignement cross-modal."""
        
        total_alignment_loss = torch.tensor(0.0, device=output.device)
        
        for modality, target in multimodal_targets.items():
            if target is not None and target.numel() > 0:
                # Adapter les dimensions
                if output.shape != target.shape:
                    min_dim = min(output.size(-1), target.size(-1))
                    output_aligned = output[..., :min_dim]
                    target_aligned = target[..., :min_dim]
                else:
                    output_aligned = output
                    target_aligned = target
                
                # Alignment loss
                alignment = F.mse_loss(output_aligned, target_aligned)
                total_alignment_loss += alignment
        
        return total_alignment_loss / len(multimodal_targets) if multimodal_targets else total_alignment_loss


class ContrastiveLoss(nn.Module):
    """Perte contrastive pour l'apprentissage de représentations."""
    
    def __init__(self, temperature: float = 0.07, negative_samples: int = 64):
        super().__init__()
        self.temperature = temperature
        self.negative_samples = negative_samples
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calcule la perte contrastive.
        
        Args:
            embeddings: Embeddings (N, D)
            labels: Labels (N,)
        """
        # Normaliser les embeddings
        embeddings = F.normalize(embeddings, dim=-1)
        
        # Calculer similarités
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Créer masque positif
        labels_expanded = labels.unsqueeze(1)
        positive_mask = (labels_expanded == labels_expanded.T).float()
        
        # Exclure diagonale
        positive_mask.fill_diagonal_(0)
        
        # InfoNCE loss
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)
        
        positive_sim = exp_sim * positive_mask
        positive_sum = positive_sim.sum(dim=1)
        
        loss = -torch.log(positive_sum / sum_exp_sim.squeeze())
        
        return loss.mean()


class MemoryReplayLoss(nn.Module):
    """Perte pour éviter l'oubli catastrophique."""
    
    def __init__(self, lambda_replay: float = 0.5):
        super().__init__()
        self.lambda_replay = lambda_replay
        self.memory_buffer = []
        self.max_memory_size = 1000
    
    def update_memory(self, samples: List[Dict[str, torch.Tensor]]):
        """Met à jour le buffer de mémoire."""
        self.memory_buffer.extend(samples)
        if len(self.memory_buffer) > self.max_memory_size:
            # Garder les plus récents
            self.memory_buffer = self.memory_buffer[-self.max_memory_size:]
    
    def forward(
        self, 
        current_output: torch.Tensor, 
        replay_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Calcule la perte de replay pour éviter l'oubli.
        """
        if not replay_outputs:
            return torch.tensor(0.0, device=current_output.device)
        
        replay_loss = torch.tensor(0.0, device=current_output.device)
        
        for replay_output in replay_outputs:
            if replay_output.shape == current_output.shape:
                # KL divergence pour maintenir les anciennes capacités
                kl_loss = F.kl_div(
                    F.log_softmax(current_output, dim=-1),
                    F.softmax(replay_output, dim=-1),
                    reduction='batchmean'
                )
                replay_loss += kl_loss
        
        return self.lambda_replay * replay_loss / len(replay_outputs) if replay_outputs else replay_loss