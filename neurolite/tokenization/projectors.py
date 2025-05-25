"""
Projecteurs et aligneurs multimodaux pour le tokenizer NeuroLite.

Ce module fournit des composants pour projeter et aligner des représentations
de différentes modalités dans un espace latent commun.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Any


class CrossModalProjector(nn.Module):
    """
    Projecteur qui fusionne des représentations de différentes modalités
    dans un espace latent commun.
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        output_dim: int,
        dropout_rate: float = 0.1
    ):
        """
        Initialise le projecteur multimodal.
        
        Args:
            input_dims: Dictionnaire des dimensions d'entrée par modalité
            output_dim: Dimension de sortie commune
            dropout_rate: Taux de dropout
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dim = output_dim
        
        # Créer les projecteurs spécifiques à chaque modalité
        self.projectors = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
            for modality, dim in input_dims.items()
        })
        
        # Couche d'attention pour pondérer les contributions des modalités
        self.modal_attention = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.GELU(),
            nn.Linear(output_dim // 2, 1)
        )
    
    def forward(self, features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Projette les caractéristiques multimodales dans un espace commun.
        
        Args:
            features_dict: Dictionnaire des caractéristiques par modalité
            
        Returns:
            Représentation unifiée dans l'espace commun
        """
        if not features_dict:
            raise ValueError("Le dictionnaire de caractéristiques est vide")
        
        # Projeter chaque modalité
        projected_features = {}
        for modality, features in features_dict.items():
            if modality not in self.projectors:
                raise ValueError(f"Modalité non prise en charge: {modality}")
            
            projected = self.projectors[modality](features)
            projected_features[modality] = projected
        
        # Si une seule modalité, renvoyer directement
        if len(projected_features) == 1:
            return list(projected_features.values())[0]
        
        # Pour plusieurs modalités, utiliser l'attention pour les fusionner
        stacked_features = torch.stack(list(projected_features.values()), dim=1)  # [B, num_modalities, output_dim]
        
        # Calculer les scores d'attention pour chaque modalité
        attention_scores = self.modal_attention(stacked_features)  # [B, num_modalities, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [B, num_modalities, 1]
        
        # Appliquer les poids d'attention
        fused_features = torch.sum(stacked_features * attention_weights, dim=1)  # [B, output_dim]
        
        return fused_features


class CrossModalAligner(nn.Module):
    """
    Aligneur multimodal qui utilise l'attention croisée pour aligner
    les représentations entre différentes modalités.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        temperature: float = 0.07
    ):
        """
        Initialise l'aligneur multimodal.
        
        Args:
            hidden_size: Dimension des représentations
            num_heads: Nombre de têtes d'attention
            dropout_rate: Taux de dropout
            temperature: Température pour l'attention
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.temperature = temperature
        
        assert hidden_size % num_heads == 0, "La dimension cachée doit être divisible par le nombre de têtes"
        self.head_dim = hidden_size // num_heads
        
        # Projections pour l'attention multi-tête
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Projecteur de contrastive learning
        self.contrastive_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Paramètre de température appris
        self.contrastive_temperature = nn.Parameter(torch.ones([]) * temperature)
    
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Applique l'attention croisée entre les requêtes et les clés/valeurs.
        
        Args:
            queries: Tenseur de requêtes [batch_size, seq_len_q, hidden_size]
            keys: Tenseur de clés [batch_size, seq_len_k, hidden_size]
            values: Tenseur de valeurs [batch_size, seq_len_k, hidden_size]
            attention_mask: Masque d'attention [batch_size, seq_len_q, seq_len_k]
            
        Returns:
            Résultat de l'attention croisée [batch_size, seq_len_q, hidden_size]
        """
        batch_size, seq_len_q, _ = queries.size()
        
        if values is None:
            values = keys
        
        # Projeter les requêtes, clés et valeurs
        q = self.q_proj(queries)
        k = self.k_proj(keys)
        v = self.v_proj(values)
        
        # Réorganiser pour l'attention multi-tête
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculer les scores d'attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Appliquer le masque d'attention si fourni
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, -1e9)
        
        # Normaliser les scores et appliquer le dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Calculer le résultat de l'attention
        context = torch.matmul(attn_weights, v)
        
        # Réorganiser et projeter
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.hidden_size)
        output = self.out_proj(context)
        
        # Connexion résiduelle et normalisation
        output = self.layer_norm(queries + output)
        
        return output
    
    def compute_contrastive_loss(
        self,
        features_dict: Dict[str, torch.Tensor],
        aligned_features_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Calcule la perte contrastive entre les modalités alignées.
        
        Args:
            features_dict: Dictionnaire des caractéristiques originales par modalité
            aligned_features_dict: Dictionnaire des caractéristiques alignées par modalité
            
        Returns:
            Perte contrastive entre les modalités
        """
        contrastive_loss = 0.0
        modalities = list(features_dict.keys())
        
        if len(modalities) <= 1:
            return torch.tensor(0.0, device=list(features_dict.values())[0].device)
        
        # Projeter les caractéristiques pour le contrastive learning
        projected_features = {
            modality: self.contrastive_projector(features)
            for modality, features in features_dict.items()
        }
        
        projected_aligned = {
            modality: self.contrastive_projector(features)
            for modality, features in aligned_features_dict.items()
        }
        
        # Normaliser les projections
        normalized_features = {
            modality: F.normalize(features, p=2, dim=-1)
            for modality, features in projected_features.items()
        }
        
        normalized_aligned = {
            modality: F.normalize(features, p=2, dim=-1)
            for modality, features in projected_aligned.items()
        }
        
        # Calculer la perte contrastive pour chaque paire de modalités
        n_pairs = 0
        for i, mod_i in enumerate(modalities):
            for j, mod_j in enumerate(modalities):
                if i < j:  # Éviter les doublons
                    # Similarité entre les caractéristiques originales
                    sim_orig = torch.matmul(
                        normalized_features[mod_i],
                        normalized_features[mod_j].transpose(-2, -1)
                    )
                    
                    # Similarité entre les caractéristiques alignées
                    sim_aligned = torch.matmul(
                        normalized_aligned[mod_i],
                        normalized_aligned[mod_j].transpose(-2, -1)
                    )
                    
                    # Calculer la perte InfoNCE
                    logits = sim_aligned / self.contrastive_temperature
                    labels = torch.arange(len(logits), device=logits.device)
                    
                    loss_i_to_j = F.cross_entropy(logits, labels)
                    loss_j_to_i = F.cross_entropy(logits.t(), labels)
                    
                    contrastive_loss += (loss_i_to_j + loss_j_to_i) / 2
                    n_pairs += 1
        
        if n_pairs > 0:
            contrastive_loss /= n_pairs
        
        return contrastive_loss
