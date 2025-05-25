"""
Fonctions de perte pour l'entraînement du tokenizer multimodal NeuroLite.

Ce module fournit différentes fonctions de perte spécialisées pour
l'entraînement du tokenizer multimodal, notamment pour l'alignement
des modalités et la quantification vectorielle.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple


class MultimodalContrastiveLoss(nn.Module):
    """
    Perte contrastive pour l'alignement des représentations multimodales.
    
    Implémente une variante de la perte InfoNCE qui encourage l'alignement
    des représentations de différentes modalités correspondant aux mêmes concepts.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialise la perte contrastive multimodale.
        
        Args:
            temperature: Température pour la normalisation softmax
        """
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
        self.cos_sim = nn.CosineSimilarity(dim=-1)
    
    def forward(
        self,
        features: torch.Tensor,
        alignments: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calcule la perte contrastive pour les caractéristiques données.
        
        Args:
            features: Caractéristiques encodées [batch_size, seq_len, hidden_size]
            alignments: Tenseur indiquant quelles caractéristiques devraient être alignées
                        Si None, utilise l'identité (chaque exemple s'aligne avec lui-même)
            
        Returns:
            Perte contrastive calculée
        """
        batch_size = features.size(0)
        
        # Normaliser les caractéristiques
        normalized_features = F.normalize(features, p=2, dim=-1)
        
        # Calculer la matrice de similarité
        sim_matrix = torch.matmul(
            normalized_features, normalized_features.transpose(-2, -1)
        ) / self.temperature
        
        # Si aucun alignement fourni, utiliser l'identité
        if alignments is None:
            # Créer des étiquettes d'identité
            labels = torch.arange(batch_size, device=features.device)
            
            # Masquer la diagonale pour éviter la trivialité
            mask = torch.eye(batch_size, device=features.device, dtype=torch.bool)
            sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        else:
            # Utiliser les alignements fournis
            labels = alignments
        
        # Calculer la perte
        loss = self.cross_entropy(sim_matrix, labels)
        
        return loss


class VQLoss(nn.Module):
    """
    Perte combinée pour la quantification vectorielle.
    
    Combine la perte de reconstruction, la perte de commitment
    et éventuellement une perte de perplexité pour encourager
    l'utilisation uniforme du codebook.
    """
    
    def __init__(
        self,
        commitment_weight: float = 0.25,
        reconstruction_weight: float = 1.0,
        perplexity_weight: float = 0.1
    ):
        """
        Initialise la perte VQ.
        
        Args:
            commitment_weight: Poids pour la perte de commitment
            reconstruction_weight: Poids pour la perte de reconstruction
            perplexity_weight: Poids pour la perte de perplexité
        """
        super().__init__()
        self.commitment_weight = commitment_weight
        self.reconstruction_weight = reconstruction_weight
        self.perplexity_weight = perplexity_weight
    
    def forward(
        self,
        inputs: torch.Tensor,
        reconstructed: torch.Tensor,
        quantized: torch.Tensor,
        codebook_indices: torch.Tensor,
        codebook_size: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calcule la perte VQ.
        
        Args:
            inputs: Entrées originales [batch_size, ..., hidden_size]
            reconstructed: Sorties reconstruites [batch_size, ..., hidden_size]
            quantized: Sorties quantifiées [batch_size, ..., hidden_size]
            codebook_indices: Indices des codes utilisés [batch_size, ...]
            codebook_size: Taille du codebook
            
        Returns:
            total_loss: Perte totale
            loss_components: Dictionnaire des composantes de perte
        """
        # Perte de reconstruction
        reconstruction_loss = F.mse_loss(reconstructed, inputs)
        
        # Perte de commitment (encourage les encodages à rester proches des codes du codebook)
        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        
        # Calcul de la perplexité (mesure l'utilisation du codebook)
        avg_probs = torch.histc(
            codebook_indices.float(),
            bins=codebook_size,
            min=0,
            max=codebook_size - 1
        ) / codebook_indices.numel()
        
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        perplexity_loss = -torch.log(perplexity / codebook_size)
        
        # Perte totale
        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            self.commitment_weight * commitment_loss +
            self.perplexity_weight * perplexity_loss
        )
        
        # Composantes de perte pour le suivi
        loss_components = {
            'reconstruction_loss': reconstruction_loss,
            'commitment_loss': commitment_loss,
            'perplexity_loss': perplexity_loss,
            'perplexity': perplexity,
            'codebook_usage': (avg_probs > 0).sum().float() / codebook_size
        }
        
        return total_loss, loss_components


class HierarchicalTokenizerLoss(nn.Module):
    """
    Perte pour l'entraînement du tokenizer hiérarchique.
    
    Combine les pertes aux différents niveaux de la hiérarchie,
    avec des poids différents selon le niveau.
    """
    
    def __init__(
        self,
        level_weights: List[float],
        contrastive_weight: float = 0.5
    ):
        """
        Initialise la perte hiérarchique.
        
        Args:
            level_weights: Poids à appliquer à chaque niveau
            contrastive_weight: Poids pour la perte contrastive
        """
        super().__init__()
        self.level_weights = level_weights
        self.contrastive_weight = contrastive_weight
        self.contrastive_loss = MultimodalContrastiveLoss()
    
    def forward(
        self,
        inputs: torch.Tensor,
        reconstructed_levels: List[torch.Tensor],
        quantized_levels: List[torch.Tensor],
        indices_levels: List[torch.Tensor],
        codebook_sizes: List[int]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calcule la perte hiérarchique.
        
        Args:
            inputs: Entrées originales
            reconstructed_levels: Sorties reconstruites à chaque niveau
            quantized_levels: Sorties quantifiées à chaque niveau
            indices_levels: Indices des codes à chaque niveau
            codebook_sizes: Tailles des codebooks à chaque niveau
            
        Returns:
            total_loss: Perte totale
            loss_components: Dictionnaire des composantes de perte
        """
        assert len(reconstructed_levels) == len(quantized_levels) == len(indices_levels) == len(codebook_sizes), \
            "Tous les niveaux doivent avoir le même nombre d'éléments"
        
        num_levels = len(reconstructed_levels)
        assert len(self.level_weights) == num_levels, \
            f"Nombre de poids ({len(self.level_weights)}) ne correspond pas au nombre de niveaux ({num_levels})"
        
        total_loss = 0.0
        loss_components = {}
        
        # Calculer la perte pour chaque niveau
        for i in range(num_levels):
            # Reconstruire l'entrée à partir de ce niveau
            reconstruction_loss = F.mse_loss(reconstructed_levels[i], inputs)
            
            # Perte de commitment
            commitment_loss = F.mse_loss(quantized_levels[i], inputs.detach())
            
            # Calculer la perplexité pour ce niveau
            avg_probs = torch.histc(
                indices_levels[i].float(),
                bins=codebook_sizes[i],
                min=0,
                max=codebook_sizes[i] - 1
            ) / indices_levels[i].numel()
            
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            
            # Perte pour ce niveau
            level_loss = (
                reconstruction_loss +
                0.25 * commitment_loss -
                0.1 * torch.log(perplexity + 1e-10)
            )
            
            # Ajouter à la perte totale avec le poids de ce niveau
            total_loss = total_loss + self.level_weights[i] * level_loss
            
            # Enregistrer les composantes
            loss_components[f'level_{i}_reconstruction'] = reconstruction_loss
            loss_components[f'level_{i}_commitment'] = commitment_loss
            loss_components[f'level_{i}_perplexity'] = perplexity
        
        # Perte contrastive entre les niveaux
        if num_levels > 1:
            contrastive_losses = []
            
            for i in range(num_levels - 1):
                for j in range(i + 1, num_levels):
                    # Calculer la perte contrastive entre les niveaux i et j
                    level_contrastive = self.contrastive_loss(
                        torch.cat([quantized_levels[i], quantized_levels[j]], dim=0)
                    )
                    contrastive_losses.append(level_contrastive)
            
            if contrastive_losses:
                avg_contrastive_loss = torch.stack(contrastive_losses).mean()
                total_loss = total_loss + self.contrastive_weight * avg_contrastive_loss
                loss_components['contrastive_loss'] = avg_contrastive_loss
        
        return total_loss, loss_components
