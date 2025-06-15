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
    Perte contrastive pour l'alignement des représentations multimodales (InfoNCE).

    Cette perte encourage les représentations de différentes modalités mais
    du même item de données à être plus proches les unes des autres que des
    représentations d'items de données différents.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initialise la perte contrastive.

        Args:
            temperature: Température pour la mise à l'échelle des logits de similarité.
        """
        super().__init__()
        self.temperature = temperature
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calcule la perte contrastive sur un dictionnaire de caractéristiques multimodales.

        Args:
            features_dict: Dictionnaire de caractéristiques encodées, où chaque valeur
                           est un tenseur de shape [batch_size, ..., hidden_dim].

        Returns:
            La perte contrastive calculée. Retourne 0 si moins de deux modalités
            sont présentes.
        """
        modalities = list(features_dict.keys())
        if len(modalities) < 2:
            return torch.tensor(0.0, device=next(iter(features_dict.values())).device)

        # Agréger les features (par ex. par la moyenne) pour obtenir un vecteur par item
        # Note : On suppose que la première dimension est le batch
        aggregated_features = {
            mod: torch.mean(feat, dim=1) for mod, feat in features_dict.items()
        }

        # Concaténer les features de toutes les modalités en un grand tenseur
        all_features = torch.cat(list(aggregated_features.values()), dim=0)
        all_features = F.normalize(all_features, p=2, dim=1) # [N * B, D]

        # Calculer la matrice de similarité cosinus
        similarity_matrix = torch.matmul(all_features, all_features.T) / self.temperature

        # Créer les étiquettes pour la perte contrastive
        batch_size = list(features_dict.values())[0].shape[0]
        num_modalities = len(modalities)
        
        labels = torch.arange(batch_size, device=all_features.device)
        labels = labels.repeat(num_modalities) # [0,1,2, 0,1,2, ...]

        # La perte est calculée en comparant chaque modalité avec toutes les autres
        total_loss = self.cross_entropy_loss(similarity_matrix, labels)
        
        return total_loss


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
                        {f'level_{i}': quantized_levels[i], f'level_{j}': quantized_levels[j]})
                    contrastive_losses.append(level_contrastive)
            
            if contrastive_losses:
                avg_contrastive_loss = torch.stack(contrastive_losses).mean()
                total_loss = total_loss + self.contrastive_weight * avg_contrastive_loss
                loss_components['contrastive_loss'] = avg_contrastive_loss
        
        return total_loss, loss_components
