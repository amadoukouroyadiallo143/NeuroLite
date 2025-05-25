"""
Encodeur spécialisé pour les entrées vidéo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union

from .image_encoder import ImageEncoder


class VideoEncoder(nn.Module):
    """
    Encodeur pour les entrées vidéo.
    Combine un encodeur d'images par trame avec un traitement temporel.
    """
    
    def __init__(
        self,
        output_dim: int,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        image_encoder_depth: int = 12,
        temporal_depth: int = 4,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.1,
        attn_dropout_rate: float = 0.0,
        temporal_model: str = "transformer",
        max_frames: int = 32,
        temporal_stride: int = 1,
        frame_pooling_method: str = "cls",
        temporal_pooling_method: str = "mean"
    ):
        """
        Initialise l'encodeur vidéo.
        
        Args:
            output_dim: Dimension de sortie
            image_size: Taille des images (supposées carrées)
            patch_size: Taille des patches (supposés carrés)
            in_channels: Nombre de canaux d'entrée (3 pour RGB)
            embed_dim: Dimension des embeddings
            image_encoder_depth: Nombre de couches transformer pour l'encodeur d'image
            temporal_depth: Nombre de couches transformer pour le traitement temporel
            num_heads: Nombre de têtes d'attention
            mlp_ratio: Ratio pour la dimension du MLP
            dropout_rate: Taux de dropout
            attn_dropout_rate: Taux de dropout pour l'attention
            temporal_model: Type de modèle temporel ("transformer", "lstm", "gru")
            max_frames: Nombre maximum de trames
            temporal_stride: Stride pour l'échantillonnage des trames
            frame_pooling_method: Méthode de pooling pour les trames ("mean", "max", "cls")
            temporal_pooling_method: Méthode de pooling temporel ("mean", "max", "last")
        """
        super().__init__()
        
        self.output_dim = output_dim
        self.max_frames = max_frames
        self.temporal_stride = temporal_stride
        self.temporal_model = temporal_model
        self.temporal_pooling_method = temporal_pooling_method
        
        # Encodeur d'image (pour traiter chaque trame individuellement)
        self.image_encoder = ImageEncoder(
            output_dim=embed_dim,  # Pas de projection finale pour l'instant
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=image_encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            use_cls_token=True,
            pooling_method=frame_pooling_method
        )
        
        # Encodage temporel (pour intégrer la dimension temporelle)
        if temporal_model == "transformer":
            # Embeddings positionnels temporels
            self.temporal_pos_embed = nn.Parameter(torch.zeros(1, max_frames, embed_dim))
            nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
            
            # Dropout
            self.pos_drop = nn.Dropout(p=dropout_rate)
            
            # Transformer pour le traitement temporel
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout_rate,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
            
            self.temporal_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=temporal_depth
            )
        
        elif temporal_model in ["lstm", "gru"]:
            # RNN bidirectionnelle
            rnn_class = nn.LSTM if temporal_model == "lstm" else nn.GRU
            self.temporal_encoder = rnn_class(
                input_size=embed_dim,
                hidden_size=embed_dim // 2,  # divisé par 2 car bidirectionnel
                num_layers=temporal_depth,
                batch_first=True,
                bidirectional=True,
                dropout=dropout_rate if temporal_depth > 1 else 0
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
        if self.temporal_stride > 1:
            indices = torch.arange(0, num_frames, self.temporal_stride, device=x.device)
            x = x[:, indices]
            if attention_mask is not None:
                attention_mask = attention_mask[:, indices]
        
        if x.shape[1] > self.max_frames:
            x = x[:, :self.max_frames]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_frames]
        
        num_frames = min(num_frames, self.max_frames)
        
        # Fusionner batch et frames pour traitement par l'encodeur d'image
        b, f, c, h, w = x.shape
        x = x.reshape(b * f, c, h, w)
        
        # Obtenir la taille d'image attendue par l'encodeur d'image
        expected_size = getattr(self.image_encoder, 'image_size', 224)
        
        # Redimensionner automatiquement si nécessaire
        if h != expected_size or w != expected_size:
            import torch.nn.functional as F
            x = F.interpolate(x, size=(expected_size, expected_size), mode='bilinear', align_corners=False)
            print(f"Redimensionnement automatique des frames vidéo de {h}x{w} à {expected_size}x{expected_size}")
        
        # Encoder chaque trame individuellement
        x = self.image_encoder(x)  # [b*f, embed_dim]
        
        # Reformer le batch avec dimension temporelle
        x = x.reshape(b, f, -1)  # [b, f, embed_dim]
        
        # Traitement temporel
        if self.temporal_model == "transformer":
            # Ajouter les embeddings positionnels temporels
            seq_len = x.size(1)
            x = x + self.temporal_pos_embed[:, :seq_len, :]
            x = self.pos_drop(x)
            
            # Encodage transformer
            if attention_mask is not None:
                # Convertir le masque d'attention au format requis par TransformerEncoder
                padding_mask = (attention_mask == 0)
                x = self.temporal_encoder(x, src_key_padding_mask=padding_mask)
            else:
                x = self.temporal_encoder(x)
            
            # Pooling temporel
            if self.temporal_pooling_method == "mean":
                if attention_mask is not None:
                    x = (x * attention_mask.unsqueeze(-1)).sum(dim=1)
                    x = x / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                else:
                    x = x.mean(dim=1)
            elif self.temporal_pooling_method == "max":
                if attention_mask is not None:
                    masked_x = x.clone()
                    masked_x[~attention_mask.bool().unsqueeze(-1)] = -1e9
                    x = masked_x.max(dim=1)[0]
                else:
                    x = x.max(dim=1)[0]
            elif self.temporal_pooling_method == "last":
                if attention_mask is not None:
                    # Prendre la dernière trame non masquée
                    last_indices = attention_mask.sum(dim=1, keepdim=True).long() - 1
                    last_indices = last_indices.clamp(min=0)
                    x = x.gather(1, last_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))).squeeze(1)
                else:
                    x = x[:, -1]
        
        elif self.temporal_model in ["lstm", "gru"]:
            # Traitement RNN
            if attention_mask is not None:
                # Créer un pack_padded_sequence pour le RNN
                lengths = attention_mask.sum(dim=1).cpu()
                x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False
                )
            
            # Passage dans le RNN
            if self.temporal_model == "lstm":
                x, (hidden, _) = self.temporal_encoder(x)
            else:  # GRU
                x, hidden = self.temporal_encoder(x)
            
            if attention_mask is not None:
                # Décompresser le résultat
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            
            # Pooling
            if self.temporal_pooling_method == "mean":
                if attention_mask is not None:
                    x = (x * attention_mask.unsqueeze(-1)).sum(dim=1)
                    x = x / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                else:
                    x = x.mean(dim=1)
            elif self.temporal_pooling_method == "max":
                if attention_mask is not None:
                    masked_x = x.clone()
                    masked_x[~attention_mask.bool().unsqueeze(-1)] = -1e9
                    x = masked_x.max(dim=1)[0]
                else:
                    x = x.max(dim=1)[0]
            elif self.temporal_pooling_method == "last":
                # Utiliser le dernier état caché
                if isinstance(hidden, tuple):  # LSTM
                    hidden = hidden[0]
                # Combiner les directions (2, batch, hidden) -> (batch, hidden*2)
                x = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # Projection finale
        x = self.output_projection(x)
        
        return x
