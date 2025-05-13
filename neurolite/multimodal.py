"""
Module de projection multimodale pour NeuroLite.
Permet d'intégrer des entrées de différentes modalités (texte, image, audio)
dans l'espace de représentation commun de NeuroLite.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
from .projection import MinHashBloomProjection

class MultimodalProjection(nn.Module):
    """
    Projection multimodale pour NeuroLite.
    Convertit différentes modalités d'entrée en représentations vectorielles
    compatibles avec l'architecture NeuroLite.
    """
    
    def __init__(
        self,
        output_dim: int,
        minhash_permutations: int,
        bloom_filter_size: int,
        image_patch_size: int = 16,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.output_dim = output_dim
        
        # Projecteur pour le texte basé sur MinHash et Bloom
        self.text_encoder = MinHashBloomProjection(
            output_dim=output_dim,
            minhash_permutations=minhash_permutations,
            bloom_filter_size=bloom_filter_size,
            dropout_rate=dropout_rate
        )
        
        # Encodeur d'images allégé (inspiré des ViT minimalistes)
        self.image_encoder = nn.Sequential(
            # Convertir l'image en patches et les projeter
            nn.Conv2d(3, 16, kernel_size=image_patch_size, stride=image_patch_size),
            nn.LayerNorm([16, 14, 14]),  # Pour images 224x224
            nn.GELU(),
            nn.Flatten(1),  # [B, 16*14*14]
            nn.Linear(16*14*14, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Encodeur audio simplifié (basé sur des features spectrales)
        self.audio_encoder = nn.Sequential(
            # Expectation: Mel-spectrogramme prétraité [B, 1, T, F]
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([16, 64, 40]),  # Pour spectrogrammes standard
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.LayerNorm([32, 15, 9]),
            nn.GELU(),
            nn.Flatten(1),
            nn.Linear(32*15*9, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Fusion adaptative des modalités
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3.0)  # Égalité par défaut
        self.fusion_gate = nn.Sequential(
            nn.Linear(output_dim * 3, 3),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, inputs: Dict[str, Union[List[str], torch.Tensor]]) -> torch.Tensor:
        """
        Traite des entrées multimodales et produit une représentation commune.
        
        Args:
            inputs: Dictionnaire avec les clés 'text', 'image', 'audio'
                   contenant les données pour chaque modalité
                   
        Returns:
            Tensor de représentations [batch_size, output_dim]
        """
        batch_size = self._get_batch_size(inputs)
        device = self._get_device(inputs)
        
        # Initialiser les représentations par modalité (tenseurs nuls si non présents)
        text_repr = torch.zeros((batch_size, self.output_dim), device=device)
        image_repr = torch.zeros((batch_size, self.output_dim), device=device)
        audio_repr = torch.zeros((batch_size, self.output_dim), device=device)
        
        # Encoder chaque modalité si présente
        if "text" in inputs and inputs["text"]:
            text_repr = self.text_encoder(inputs["text"])
            
        if "image" in inputs and inputs["image"] is not None:
            image_repr = self.image_encoder(inputs["image"])
            
        if "audio" in inputs and inputs["audio"] is not None:
            audio_repr = self.audio_encoder(inputs["audio"])
        
        # Fusion adaptative par gate mechanism
        combined_repr = torch.cat([text_repr, image_repr, audio_repr], dim=-1)
        fusion_weights = self.fusion_gate(combined_repr)
        
        # Calculer une somme pondérée des représentations
        fused_repr = (
            fusion_weights[:, 0:1] * text_repr +
            fusion_weights[:, 1:2] * image_repr +
            fusion_weights[:, 2:3] * audio_repr
        )
        
        return fused_repr
    
    def _get_batch_size(self, inputs: Dict[str, Union[List[str], torch.Tensor]]) -> int:
        """Détermine la taille du batch à partir des entrées"""
        if "text" in inputs and inputs["text"]:
            return len(inputs["text"])
        elif "image" in inputs and inputs["image"] is not None:
            return inputs["image"].size(0)
        elif "audio" in inputs and inputs["audio"] is not None:
            return inputs["audio"].size(0)
        else:
            return 1  # Par défaut
            
    def _get_device(self, inputs: Dict[str, Union[List[str], torch.Tensor]]) -> torch.device:
        """Détermine le device des tenseurs d'entrée"""
        if "image" in inputs and inputs["image"] is not None:
            return inputs["image"].device
        elif "audio" in inputs and inputs["audio"] is not None:
            return inputs["audio"].device
        else:
            return torch.device("cpu")  # Par défaut


class CrossModalAttention(nn.Module):
    """
    Module d'attention cross-modale pour fusionner des informations
    entre différentes modalités.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        # Projections pour l'attention
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(
        self,
        query_modality: torch.Tensor,
        key_value_modality: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Applique l'attention cross-modale.
        
        Args:
            query_modality: Tensor de la modalité de requête [batch, seq_len_q, hidden_size]
            key_value_modality: Tensor de la modalité clé/valeur [batch, seq_len_kv, hidden_size]
            attention_mask: Masque d'attention optionnel [batch, seq_len_q, seq_len_kv]
            
        Returns:
            Tensor fusionné [batch, seq_len_q, hidden_size]
        """
        residual = query_modality
        
        batch_size, seq_len_q, _ = query_modality.shape
        _, seq_len_kv, _ = key_value_modality.shape
        
        # Projections
        q = self.q_proj(query_modality).view(
            batch_size, seq_len_q, self.num_heads, self.head_size
        ).transpose(1, 2)  # [batch, heads, seq_len_q, head_size]
        
        k = self.k_proj(key_value_modality).view(
            batch_size, seq_len_kv, self.num_heads, self.head_size
        ).transpose(1, 2)  # [batch, heads, seq_len_kv, head_size]
        
        v = self.v_proj(key_value_modality).view(
            batch_size, seq_len_kv, self.num_heads, self.head_size
        ).transpose(1, 2)  # [batch, heads, seq_len_kv, head_size]
        
        # Calcul de l'attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_size ** 0.5)
        
        if attention_mask is not None:
            # Appliquer le masque (ajouter -inf où le masque est 0)
            scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, float("-inf"))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Appliquer l'attention aux valeurs
        context = torch.matmul(attn_weights, v)  # [batch, heads, seq_len_q, head_size]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.hidden_size
        )  # [batch, seq_len_q, hidden_size]
        
        # Projection de sortie et connexion résiduelle
        output = self.output_proj(context)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)
        
        return output
