"""
Encodeur spécialisé pour les entrées d'images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Union
import math

class PatchEmbedding(nn.Module):
    """
    Module de découpage d'image en patches et projection.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        use_conv: bool = True,
        layer_norm: bool = True
    ):
        """
        Initialise le module d'embedding de patches.
        
        Args:
            image_size: Taille de l'image (supposée carrée)
            patch_size: Taille des patches (supposés carrés)
            in_channels: Nombre de canaux d'entrée (3 pour RGB)
            embed_dim: Dimension d'embedding
            use_conv: Si True, utilise la convolution pour l'embedding
            layer_norm: Si True, applique la normalisation de couche
        """
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        self.num_patches = (image_size // patch_size) ** 2
        
        if use_conv:
            # Utiliser une convolution pour découper et projeter
            self.projection = nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            )
        else:
            # Utiliser une approche de reshaping + projection linéaire
            self.projection = nn.Sequential(
                nn.Unfold(kernel_size=patch_size, stride=patch_size),
                nn.Linear(in_channels * patch_size * patch_size, embed_dim)
            )
        
        # Normalisation optionnelle
        self.layer_norm = nn.LayerNorm(embed_dim) if layer_norm else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Découpe l'image en patches et les projette.
        
        Args:
            x: Images d'entrée [batch_size, in_channels, height, width]
            
        Returns:
            Embeddings des patches [batch_size, num_patches, embed_dim]
        """
        batch_size, channels, height, width = x.shape
        
        # Vérifier les dimensions
        assert height == width == self.image_size, \
            f"Input image size ({height}*{width}) doesn't match expected size ({self.image_size}*{self.image_size})"
        assert channels == self.in_channels, \
            f"Input channels ({channels}) don't match expected channels ({self.in_channels})"
        
        # Projeter les patches
        if isinstance(self.projection, nn.Conv2d):
            # Approche par convolution
            x = self.projection(x)  # [B, embed_dim, grid_size, grid_size]
            x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        else:
            # Approche par reshaping + projection linéaire
            x = self.projection(x)  # [B, embed_dim*num_patches]
            x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Normalisation
        x = self.layer_norm(x)
        
        return x


class ImageEncoder(nn.Module):
    """
    Encodeur pour les entrées d'images, basé sur une architecture Vision Transformer.
    """
    
    def __init__(
        self,
        output_dim: int,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.1,
        attn_dropout_rate: float = 0.0,
        use_cls_token: bool = True,
        pooling_method: str = "cls"
    ):
        """
        Initialise l'encodeur d'images.
        
        Args:
            output_dim: Dimension de sortie
            image_size: Taille de l'image (supposée carrée)
            patch_size: Taille des patches (supposés carrés)
            in_channels: Nombre de canaux d'entrée (3 pour RGB)
            embed_dim: Dimension des embeddings
            depth: Nombre de couches transformer
            num_heads: Nombre de têtes d'attention
            mlp_ratio: Ratio pour la dimension du MLP
            dropout_rate: Taux de dropout
            attn_dropout_rate: Taux de dropout pour l'attention
            use_cls_token: Si True, ajoute un token CLS pour le pooling
            pooling_method: Méthode de pooling ("mean", "max", "cls")
        """
        super().__init__()
        
        self.output_dim = output_dim
        self.use_cls_token = use_cls_token
        self.pooling_method = pooling_method
        self.embed_dim = embed_dim
        
        # Embedding des patches
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Nombre de patches
        self.num_patches = self.patch_embed.num_patches
        
        # Embeddings positionnels
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + (1 if use_cls_token else 0), embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Token CLS
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Dropout
        self.pos_drop = nn.Dropout(p=dropout_rate)
        
        # Couches Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=depth
        )
        
        # Projection de sortie
        self.output_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )
        
        # Initialisation des poids
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialise les poids du modèle."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode des images.
        
        Args:
            x: Images d'entrée [batch_size, in_channels, height, width]
            
        Returns:
            Représentation encodée [batch_size, output_dim]
        """
        batch_size = x.shape[0]
        
        # Découper l'image en patches et projeter
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Ajouter le token CLS si nécessaire
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, embed_dim]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, 1 + num_patches, embed_dim]
        
        # Ajouter les embeddings positionnels
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Pooling
        if self.pooling_method == "cls" and self.use_cls_token:
            x = x[:, 0]  # Prendre le token CLS
        elif self.pooling_method == "mean":
            if self.use_cls_token:
                x = x[:, 1:].mean(dim=1)  # Exclure le token CLS
            else:
                x = x.mean(dim=1)
        elif self.pooling_method == "max":
            if self.use_cls_token:
                x = x[:, 1:].max(dim=1)[0]  # Exclure le token CLS
            else:
                x = x.max(dim=1)[0]
        
        # Projection finale
        x = self.output_projection(x)
        
        return x
