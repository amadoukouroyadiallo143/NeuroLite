"""
NeuroLite Multimodal Fusion - Version Nettoyée v2.0
===================================================

⚠️ FICHIER NETTOYÉ - Contient seulement les classes utilisées
Classes supprimées : ModalityEncoder, MultimodalFusionCenter (obsolètes)
Nouveau système : voir core/super_multimodal_processor.py

Classes conservées pour SuperMultimodalProcessor :
- FusionStrategy, UnifiedRepresentation  
- CrossModalAttention, HierarchicalFusion, AdaptiveFusionGate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time

from .brain_architecture import BrainSignal, BrainRegion

class FusionStrategy(Enum):
    """Stratégies de fusion multimodale"""
    EARLY_FUSION = "early_fusion"          # Fusion au début
    LATE_FUSION = "late_fusion"            # Fusion à la fin
    HIERARCHICAL_FUSION = "hierarchical"   # Fusion hiérarchique
    ATTENTION_FUSION = "attention"         # Fusion par attention
    ADAPTIVE_FUSION = "adaptive"           # Fusion adaptative

@dataclass
class UnifiedRepresentation:
    """Représentation unifiée de tous les signaux multimodaux"""
    unified_tensor: torch.Tensor
    modality_weights: Dict[str, float]
    confidence_score: float
    source_signals: List[BrainSignal]
    fusion_metadata: Dict[str, Any]
    timestamp: float

class CrossModalAttention(nn.Module):
    """Module d'attention croisée entre modalités"""
    
    def __init__(self, embed_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Projections pour Q, K, V
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Projection de sortie
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Normalisation
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass de l'attention croisée"""
        
        batch_size, seq_len = query.shape[:2]
        
        # Projections
        Q = self.query_proj(query)  # [B, Lq, D]
        K = self.key_proj(key)      # [B, Lk, D]
        V = self.value_proj(value)  # [B, Lv, D]
        
        # Reshape pour multi-head
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Appliquer l'attention
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape et projection finale
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim
        )
        output = self.out_proj(attn_output)
        
        # Connexion résiduelle et normalisation
        output = self.norm1(query + self.dropout(output))
        
        # FFN avec connexion résiduelle
        ffn_output = self.ffn(output)
        output = self.norm2(output + self.dropout(ffn_output))
        
        return output

# ❌ ModalityEncoder SUPPRIMÉ - Non utilisé par SuperMultimodalProcessor
# Les encodeurs sont maintenant intégrés directement dans SuperMultimodalProcessor

class HierarchicalFusion(nn.Module):
    """Fusion hiérarchique des modalités"""
    
    def __init__(self, embed_dim: int = 512, num_levels: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        
        # Couches de fusion par niveau
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(num_levels)
        ])
        
        # Attention pour chaque niveau
        self.level_attentions = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
            for _ in range(num_levels)
        ])
        
        # Fusion finale
        self.final_fusion = nn.Sequential(
            nn.Linear(embed_dim * num_levels, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    
    def forward(self, modality_features: torch.Tensor) -> torch.Tensor:
        """Fusion hiérarchique"""
        level_outputs = []
        current_features = modality_features
        
        for level in range(self.num_levels):
            # Fusion au niveau actuel
            fused = self.fusion_layers[level](current_features)
            
            # Attention au niveau
            if fused.dim() == 2:
                fused = fused.unsqueeze(0)  # Add batch dim if needed
            
            attended, _ = self.level_attentions[level](fused, fused, fused)
            level_output = attended.mean(dim=1) if attended.dim() == 3 else attended
            
            level_outputs.append(level_output)
            current_features = level_output
        
        # Fusion finale de tous les niveaux
        if level_outputs:
            concatenated = torch.cat(level_outputs, dim=-1)
            final_output = self.final_fusion(concatenated)
        else:
            final_output = torch.zeros(1, self.embed_dim)
        
        return final_output

class AdaptiveFusionGate(nn.Module):
    """Porte de fusion adaptative qui apprend les poids optimaux"""
    
    def __init__(self, num_modalities: int, embed_dim: int = 512):
        super().__init__()
        self.num_modalities = num_modalities
        self.embed_dim = embed_dim
        
        # Réseau pour calculer les poids de fusion (adaptatif)
        # Ne spécifie pas la taille d'entrée, sera créé dynamiquement
        self.fusion_network = None
        self._fusion_input_size = None
        
        # Réseau de confiance
        self.confidence_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, modality_features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fusion adaptative avec poids appris"""
        if not modality_features:
            return torch.zeros(1, self.embed_dim), torch.zeros(1), torch.zeros(self.num_modalities)
        
        # Normaliser les dimensions
        normalized_features = []
        for feat in modality_features:
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            if feat.size(-1) != self.embed_dim:
                # Adapter la dimension si nécessaire
                if feat.size(-1) < self.embed_dim:
                    padding = torch.zeros(feat.size(0), self.embed_dim - feat.size(-1))
                    feat = torch.cat([feat, padding], dim=-1)
                else:
                    feat = feat[:, :self.embed_dim]
            normalized_features.append(feat)
        
        # Calculer les poids de fusion de manière adaptative
        concat_features = torch.cat(normalized_features, dim=-1)
        
        # Créer le réseau dynamiquement si nécessaire
        if self.fusion_network is None or self._fusion_input_size != concat_features.size(-1):
            self._fusion_input_size = concat_features.size(-1)
            self.fusion_network = nn.Sequential(
                nn.Linear(self._fusion_input_size, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, len(normalized_features)),  # Nombre réel de modalités
                nn.Softmax(dim=-1)
            ).to(concat_features.device)
        
        fusion_weights = self.fusion_network(concat_features)
        
        # Fusion pondérée
        fused_output = torch.zeros_like(normalized_features[0])
        for i, feat in enumerate(normalized_features):
            weight = fusion_weights[:, i:i+1]
            fused_output += weight * feat
        
        # Calculer la confiance
        confidence = self.confidence_network(fused_output)
        
        return fused_output, confidence.squeeze(), fusion_weights.squeeze()

# ❌ MultimodalFusionCenter SUPPRIMÉ - Remplacé par SuperMultimodalProcessor
# Nouveau système dans core/super_multimodal_processor.py avec :
# - Tokenization universelle intégrée  
# - 6 modalités vs 6 modalités anciennes
# - Cache multiniveau intelligent
# - Traitement parallèle optimisé

# ============ ALIAS DE COMPATIBILITÉ ============

# ⚠️ ALIAS SUPPRIMÉS - Maintenant définis dans super_multimodal_processor.py
# UnifiedMultimodalFusion = SuperMultimodalProcessor
# EnhancedMultimodalFusionCenter = SuperMultimodalProcessor

# 🎯 UTILISER MAINTENANT : from .super_multimodal_processor import SuperMultimodalProcessor