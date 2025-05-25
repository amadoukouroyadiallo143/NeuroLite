"""
Tokenizer hiérarchique pour l'architecture NeuroLite.

Ce module implémente un tokenizer hiérarchique qui préserve les
relations structurelles entre les tokens à différents niveaux d'abstraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Any

from .quantizers import HierarchicalVQ


class RelationPredictor(nn.Module):
    """
    Prédit les relations entre tokens dans un espace hiérarchique.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_relation_types: int
    ):
        """
        Initialise le prédicteur de relations.
        
        Args:
            input_dim: Dimension des vecteurs d'entrée
            hidden_dim: Dimension cachée
            num_relation_types: Nombre de types de relations à prédire
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_relation_types = num_relation_types
        
        # Réseau pour prédire les relations
        self.relation_predictor = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_relation_types)
        )
    
    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Prédit les relations entre tous les pairs de tokens.
        
        Args:
            tokens: Tokens à analyser [batch_size, seq_len, input_dim]
            mask: Masque binaire indiquant les tokens valides [batch_size, seq_len]
            
        Returns:
            relations: Probabilités des relations [batch_size, seq_len, seq_len, num_relation_types]
        """
        batch_size, seq_len, _ = tokens.size()
        
        # Créer toutes les paires possibles de tokens
        token_i = tokens.unsqueeze(2).expand(batch_size, seq_len, seq_len, self.input_dim)
        token_j = tokens.unsqueeze(1).expand(batch_size, seq_len, seq_len, self.input_dim)
        
        # Concaténer les paires
        token_pairs = torch.cat([token_i, token_j], dim=-1)
        
        # Prédire les relations
        relations_logits = self.relation_predictor(token_pairs)
        relations_probs = F.softmax(relations_logits, dim=-1)
        
        # Appliquer le masque si fourni
        if mask is not None:
            mask_i = mask.unsqueeze(2).expand(batch_size, seq_len, seq_len)
            mask_j = mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
            pair_mask = mask_i & mask_j
            relations_probs = relations_probs * pair_mask.unsqueeze(-1)
        
        return relations_probs


class HierarchicalTokenizer(nn.Module):
    """
    Tokenizer hiérarchique qui préserve les relations structurelles.
    """
    
    def __init__(
        self,
        level_dims: List[int],
        level_vocab_sizes: List[int],
        level_commitment_costs: List[float],
        relation_hidden_dim: int,
        num_relation_types: int,
        use_ema_updates: bool = True,
        ema_decay: float = 0.99
    ):
        """
        Initialise le tokenizer hiérarchique.
        
        Args:
            level_dims: Dimensions pour chaque niveau
            level_vocab_sizes: Tailles de vocabulaire pour chaque niveau
            level_commitment_costs: Coûts de commitment pour chaque niveau
            relation_hidden_dim: Dimension cachée pour le prédicteur de relations
            num_relation_types: Nombre de types de relations à prédire
            use_ema_updates: Si True, utilise EMA pour les mises à jour
            ema_decay: Taux de decay pour EMA
        """
        super().__init__()
        
        self.num_levels = len(level_dims)
        
        # Quantificateur vectoriel hiérarchique
        self.hierarchical_vq = HierarchicalVQ(
            level_dims=level_dims,
            level_codebook_sizes=level_vocab_sizes,
            level_commitment_costs=level_commitment_costs,
            use_ema_updates=use_ema_updates,
            ema_decay=ema_decay
        )
        
        # Projecteurs pour la reconstruction
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(level_dims[i], level_dims[0]),
                nn.LayerNorm(level_dims[0]),
                nn.GELU(),
                nn.Linear(level_dims[0], level_dims[0])
            )
            for i in range(self.num_levels)
        ])
        
        # Prédicteurs de relations entre tokens
        self.intra_level_relation_predictors = nn.ModuleList([
            RelationPredictor(
                input_dim=level_dims[i],
                hidden_dim=relation_hidden_dim,
                num_relation_types=num_relation_types
            )
            for i in range(self.num_levels)
        ])
        
        # Prédicteurs de relations entre niveaux
        self.inter_level_relation_predictors = nn.ModuleList([
            RelationPredictor(
                input_dim=level_dims[i] + level_dims[i+1],
                hidden_dim=relation_hidden_dim,
                num_relation_types=num_relation_types
            )
            for i in range(self.num_levels - 1)
        ])
    
    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Tokenize les features de façon hiérarchique avec relations.
        
        Args:
            features: Caractéristiques à tokenizer [batch_size, seq_len, level_dims[0]]
            attention_mask: Masque d'attention [batch_size, seq_len]
            
        Returns:
            Dictionnaire contenant les tokens, indices et relations
        """
        # Quantification vectorielle hiérarchique
        quantized_levels, indices_levels, commitment_loss = self.hierarchical_vq(features)
        
        # Reconstruire les entrées à partir de chaque niveau
        reconstructed_levels = [
            self.decoders[i](quantized)
            for i, quantized in enumerate(quantized_levels)
        ]
        
        # Prédire les relations intra-niveau
        intra_level_relations = []
        for i, quantized in enumerate(quantized_levels):
            relations = self.intra_level_relation_predictors[i](
                quantized, attention_mask
            )
            intra_level_relations.append(relations)
        
        # Prédire les relations inter-niveaux
        inter_level_relations = []
        for i in range(self.num_levels - 1):
            # Adapter les dimensions si nécessaire pour l'alignement des séquences
            level_i = quantized_levels[i]
            level_i_plus_1 = quantized_levels[i+1]
            
            # Si les longueurs de séquence sont différentes, interpoler
            if level_i.size(1) != level_i_plus_1.size(1):
                # Interpoler le niveau i+1 pour correspondre au niveau i
                level_i_plus_1_interpolated = F.interpolate(
                    level_i_plus_1.transpose(1, 2),
                    size=level_i.size(1),
                    mode='linear'
                ).transpose(1, 2)
            else:
                level_i_plus_1_interpolated = level_i_plus_1
            
            # Concaténer les niveaux
            combined_levels = torch.cat(
                [level_i, level_i_plus_1_interpolated],
                dim=-1
            )
            
            # Prédire les relations
            relations = self.inter_level_relation_predictors[i](
                combined_levels, attention_mask
            )
            inter_level_relations.append(relations)
        
        return {
            'quantized_levels': quantized_levels,
            'indices_levels': indices_levels,
            'reconstructed_levels': reconstructed_levels,
            'commitment_loss': commitment_loss,
            'intra_level_relations': intra_level_relations,
            'inter_level_relations': inter_level_relations
        }
    
    def decode_indices(
        self,
        indices_levels: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Décode les indices en représentations continues.
        
        Args:
            indices_levels: Liste des indices à chaque niveau
            
        Returns:
            Liste des représentations décodées à chaque niveau
        """
        # Décoder les indices en représentations quantifiées
        quantized_levels = self.hierarchical_vq.decode_indices(indices_levels)
        
        # Reconstruire à partir des représentations quantifiées
        reconstructed_levels = [
            self.decoders[i](quantized)
            for i, quantized in enumerate(quantized_levels)
        ]
        
        return reconstructed_levels
    
    def tokenize(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Interface simplifiée pour la tokenization.
        
        Args:
            features: Caractéristiques à tokenizer
            attention_mask: Masque d'attention
            
        Returns:
            indices_levels: Liste des indices de tokens à chaque niveau
            metadata: Métadonnées supplémentaires (relations, etc.)
        """
        outputs = self.forward(features, attention_mask)
        
        metadata = {
            'intra_level_relations': outputs['intra_level_relations'],
            'inter_level_relations': outputs['inter_level_relations'],
            'commitment_loss': outputs['commitment_loss']
        }
        
        return outputs['indices_levels'], metadata
