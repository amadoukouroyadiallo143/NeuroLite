"""
Encodeur spécialisé pour les entrées d'images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Union
import math

from neurolite.core.ssm import SSMLayer # Added import
from neurolite.Configs.config import MMImageEncoderConfig
from .base_encoder import BaseEncoder

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


class ImageEncoder(BaseEncoder):
    """
    Encodeur pour les entrées d'images, basé sur une architecture Vision Transformer
    ou SSM, configuré via MMImageEncoderConfig.
    """
    def __init__(self, config: MMImageEncoderConfig):
        """
        Initialise l'encodeur d'images à partir d'un objet de configuration.
        """
        super().__init__(config)
        self.config = config

        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.num_channels,
            embed_dim=config.hidden_dim
        )
        
        num_patches = self.patch_embed.num_patches
        
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + (1 if config.use_cls_token else 0), config.hidden_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        self.pos_drop = nn.Dropout(p=config.dropout_rate)
        
        if config.use_ssm:
            ssm_layers = [
                SSMLayer(
                    dim=config.hidden_dim,
                    d_state=config.ssm_d_state,
                    d_conv=config.ssm_d_conv,
                    expand_factor=config.ssm_expand_factor,
                    bidirectional=config.ssm_bidirectional
                ) for _ in range(config.num_layers)
            ]
            self.encoder_layers = nn.Sequential(*ssm_layers)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=int(config.hidden_dim * config.mlp_ratio),
                dropout=config.dropout_rate,
                activation=config.activation,
                batch_first=True,
                norm_first=True
            )
            self.encoder_layers = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=config.num_layers
            )
        
        self.output_projection = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
        
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
        if self.config.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, embed_dim]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, 1 + num_patches, embed_dim]
        
        # Ajouter les embeddings positionnels
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Encoder layers (Transformer or SSM)
        x = self.encoder_layers(x)
        
        # Pooling
        if self.config.use_cls_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
            
        # Projection finale
        output = self.output_projection(x)
        
        return output
