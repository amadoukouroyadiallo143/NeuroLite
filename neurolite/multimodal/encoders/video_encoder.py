"""
Encodeur spécialisé pour les entrées vidéo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union

from neurolite.Configs.config import MMVideoEncoderConfig, MMImageEncoderConfig
from .base_encoder import BaseEncoder
from .image_encoder import ImageEncoder
from neurolite.core.ssm import SSMLayer # Added import


class VideoEncoder(BaseEncoder):
    """
    Encodeur pour les entrées vidéo, configuré via MMVideoEncoderConfig.
    Combine un encodeur d'images par trame avec un traitement temporel.
    """
    def __init__(self, config: MMVideoEncoderConfig):
        """
        Initialise l'encodeur vidéo à partir d'un objet de configuration.
        """
        super().__init__(config)
        self.config = config

        # L'encodeur d'image interne est configuré pour sortir la `hidden_dim`
        # qui sert d'entrée à l'encodeur temporel.
        image_encoder_internal_config = config.image_encoder_config
        image_encoder_internal_config.output_dim = config.hidden_dim

        self.image_encoder = ImageEncoder(config=image_encoder_internal_config)
        
        # Embeddings positionnels temporels
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, config.num_frames_input, config.hidden_dim)
        )
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        
        self.pos_drop = nn.Dropout(p=config.dropout_rate)
        
        # Transformer ou SSM pour le traitement temporel
        if config.temporal_use_ssm:
            ssm_layers = [
                SSMLayer(
                    dim=config.hidden_dim,
                    d_state=config.ssm_d_state,
                    d_conv=config.ssm_d_conv,
                    expand_factor=config.ssm_expand_factor,
                    bidirectional=config.ssm_bidirectional
                ) for _ in range(config.temporal_num_layers)
            ]
            self.temporal_encoder = nn.Sequential(*ssm_layers)
        else:
            temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.temporal_num_attention_heads,
            dim_feedforward=int(config.hidden_dim * config.temporal_mlp_ratio),
            dropout=config.dropout_rate,
            activation=config.activation,
            batch_first=True,
            norm_first=True
            )
            self.temporal_encoder = nn.TransformerEncoder(
                    encoder_layer=temporal_encoder_layer,
                num_layers=config.temporal_num_layers
            )
        
        # Projection de sortie
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
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode des vidéos.
        
        Args:
            x: Vidéos d'entrée [batch_size, frames, channels, height, width]
               ou [batch_size, channels, frames, height, width]
            attention_mask: Masque d'attention pour les trames [batch_size, frames]
            
        Returns:
            Représentation encodée [batch_size, output_dim]
        """
        batch_size = x.shape[0]
        
        # Réorganiser l'entrée si nécessaire
        if x.shape[1] == 3:  # [B, C, F, H, W] format
            x = x.permute(0, 2, 1, 3, 4)  # -> [B, F, C, H, W]
        
        num_frames = x.shape[1]
        
        # Limiter le nombre de trames et appliquer le stride
        if self.config.temporal_stride > 1:
            indices = torch.arange(0, num_frames, self.config.temporal_stride, device=x.device)
            x = x[:, indices]
            if attention_mask is not None:
                attention_mask = attention_mask[:, indices]
        
        if x.shape[1] > self.config.num_frames_input:
            x = x[:, :self.config.num_frames_input]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.config.num_frames_input]
        
        num_frames = min(num_frames, self.config.num_frames_input)
        
        # Fusionner batch et frames pour traitement par l'encodeur d'image
        b, f, c, h, w = x.shape
        x = x.reshape(b * f, c, h, w)
        
        # Encoder chaque trame individuellement
        x = self.image_encoder(x)  # [b*f, hidden_dim]
        
        # Reformer le batch avec dimension temporelle
        x = x.reshape(b, f, -1)  # [b, f, hidden_dim]
        
        # Ajouter les embeddings positionnels temporels
        seq_len = x.size(1)
        x = x + self.temporal_pos_embed[:, :seq_len, :]
        x = self.pos_drop(x)
            
        # Encodage temporel (Transformer ou SSM)
        if attention_mask is not None:
            # Pour le Transformer, le masque est `src_key_padding_mask`
            if isinstance(self.temporal_encoder, nn.TransformerEncoder):
                padding_mask = (attention_mask == 0)
                x = self.temporal_encoder(x, src_key_padding_mask=padding_mask)
            else: # Pour le SSM ou autre, on suppose qu'il n'y a pas de masque
                x = self.temporal_encoder(x)
        else:
            x = self.temporal_encoder(x)
            
        # Pooling temporel (moyenne)
        if attention_mask is not None:
            pooled_output = (x * attention_mask.unsqueeze(-1)).sum(dim=1)
            pooled_output = pooled_output / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled_output = x.mean(dim=1)
        
        # Projection finale
        output = self.output_projection(pooled_output)
        
        return output
