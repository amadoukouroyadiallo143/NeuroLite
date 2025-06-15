"""
Décodeur spécialisé pour la génération de vidéos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union, Tuple, Any
from dataclasses import dataclass

from neurolite.Configs.config import MMVideoDecoderConfig
from .image_decoder import ImageDecoder, UpsampleBlock
from .base_decoder import BaseDecoder


class TemporalUpsampleBlock(nn.Module):
    """
    Bloc de suréchantillonnage temporel pour la génération de vidéos.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel_size: int = 3,
        spatial_kernel_size: int = 3,
        temporal_stride: int = 1,
        spatial_stride: int = 1,
        temporal_padding: int = 1,
        spatial_padding: int = 1,
        temporal_upsample: bool = True,
        spatial_upsample: bool = True,
        upsample_factor: int = 2,
        use_residual: bool = True,
        dropout_rate: float = 0.1
    ):
        """
        Initialise un bloc de suréchantillonnage temporel.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            temporal_kernel_size: Taille du noyau temporel
            spatial_kernel_size: Taille du noyau spatial
            temporal_stride: Pas temporel
            spatial_stride: Pas spatial
            temporal_padding: Padding temporel
            spatial_padding: Padding spatial
            temporal_upsample: Si True, effectue un suréchantillonnage temporel
            spatial_upsample: Si True, effectue un suréchantillonnage spatial
            upsample_factor: Facteur de suréchantillonnage
            use_residual: Si True, ajoute une connexion résiduelle
            dropout_rate: Taux de dropout
        """
        super().__init__()
        
        self.use_residual = use_residual
        self.temporal_upsample = temporal_upsample
        self.spatial_upsample = spatial_upsample
        
        # Suréchantillonnage temporel
        if temporal_upsample:
            self.temporal_up = nn.Upsample(
                scale_factor=(upsample_factor, 1, 1),
                mode="trilinear",
                align_corners=False
            )
        
        # Suréchantillonnage spatial
        if spatial_upsample:
            self.spatial_up = nn.Upsample(
                scale_factor=(1, upsample_factor, upsample_factor),
                mode="trilinear",
                align_corners=False
            )
        
        # Convolution temporelle
        self.temporal_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(temporal_kernel_size, 1, 1),
            stride=(temporal_stride, 1, 1),
            padding=(temporal_padding, 0, 0)
        )
        
        # Convolution spatiale
        self.spatial_conv = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(1, spatial_kernel_size, spatial_kernel_size),
            stride=(1, spatial_stride, spatial_stride),
            padding=(0, spatial_padding, spatial_padding)
        )
        
        # Normalisation et activation
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout3d(dropout_rate)
        
        # Projection résiduelle si nécessaire
        if use_residual:
            layers = []
            if temporal_upsample or spatial_upsample:
                scale_factor = (
                    upsample_factor if temporal_upsample else 1,
                    upsample_factor if spatial_upsample else 1,
                    upsample_factor if spatial_upsample else 1
                )
                layers.append(nn.Upsample(
                    scale_factor=scale_factor,
                    mode="trilinear",
                    align_corners=False
                ))
            
            if in_channels != out_channels:
                layers.append(nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ))
            
            self.residual_proj = nn.Sequential(*layers) if layers else nn.Identity()
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applique le bloc de suréchantillonnage temporel.
        
        Args:
            x: Tenseur d'entrée [batch_size, in_channels, time, height, width]
            
        Returns:
            Tenseur traité [batch_size, out_channels, time', height', width']
        """
        identity = x
        
        # Suréchantillonnage temporel si nécessaire
        if self.temporal_upsample:
            x = self.temporal_up(x)
        
        # Convolution temporelle
        x = self.temporal_conv(x)
        x = self.norm1(x)
        x = self.act(x)
        
        # Suréchantillonnage spatial si nécessaire
        if self.spatial_upsample:
            x = self.spatial_up(x)
        
        # Convolution spatiale
        x = self.spatial_conv(x)
        x = self.norm2(x)
        
        # Connexion résiduelle
        if self.use_residual:
            identity = self.residual_proj(identity)
            x = x + identity
        
        # Activation et dropout
        x = self.act(x)
        x = self.dropout(x)
        
        return x


class VideoDecoder(BaseDecoder):
    """
    Décode une représentation latente pour générer une séquence d'images (vidéo).
    """
    def __init__(self, config: MMVideoDecoderConfig):
        """
        Initialise le décodeur vidéo.
        
        Args:
            config (MMVideoDecoderConfig): La configuration pour le décodeur vidéo.
        """
        super().__init__(config)
        self.config = config
        
        # Projection initiale du vecteur latent
        self.initial_projection = nn.Sequential(
            nn.Linear(config.input_dim, config.initial_channels * config.initial_time * config.initial_size * config.initial_size),
            nn.Dropout(config.dropout_rate),
            nn.GELU()
        )
        
        # Blocs de suréchantillonnage
        self.upsamples = nn.ModuleList()
        current_channels = config.initial_channels
        
        # Déterminer le nombre maximum d'itérations
        num_iterations = max(config.num_temporal_upsamples, config.num_spatial_upsamples)
        
        for i in range(num_iterations):
            out_channels = max(current_channels // 2, 16)
            
            # Déterminer si un suréchantillonnage temporel/spatial est nécessaire à cette étape
            temporal_upsample = i < config.num_temporal_upsamples
            spatial_upsample = i < config.num_spatial_upsamples

            self.upsamples.append(
                    TemporalUpsampleBlock(
                        in_channels=current_channels,
                        out_channels=out_channels,
                    temporal_upsample=temporal_upsample,
                    spatial_upsample=spatial_upsample,
                    upsample_factor=2,
                    use_residual=config.use_residual,
                    dropout_rate=config.dropout_rate
                )
            )
            current_channels = out_channels
            
            # Couche de sortie
            self.output_conv = nn.Conv3d(
                in_channels=current_channels,
            out_channels=config.output_channels,
            kernel_size=3,
            padding=1
            )
            
            # Activation finale
        if config.final_activation == "tanh":
                self.final_activation = nn.Tanh()
        elif config.final_activation == "sigmoid":
                self.final_activation = nn.Sigmoid()
        else:
                self.final_activation = nn.Identity()
        
        # Décodeur spatial (par image)
        if config.use_ssm_layers:
            self.spatial_decoder = nn.Sequential(*[
                SSMLayer(
                    dim=config.embedding_dim,
                    d_state=config.ssm_d_state,
                    d_conv=config.ssm_d_conv,
                    expand_factor=config.ssm_expand_factor
                ) for _ in range(config.num_spatial_layers)
            ])
        else:
            spatial_decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim,
                dropout=config.dropout_rate,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
            self.spatial_decoder = nn.TransformerDecoder(
                decoder_layer=spatial_decoder_layer,
                num_layers=config.num_spatial_layers
            )

        # Décodeur temporel (à travers les images)
        if config.use_ssm_layers:
            self.temporal_decoder = nn.Sequential(*[
                SSMLayer(
                    dim=config.embedding_dim,
                    d_state=config.ssm_d_state,
                    d_conv=config.ssm_d_conv,
                    expand_factor=config.ssm_expand_factor
                ) for _ in range(config.num_temporal_layers)
            ])
        else:
            temporal_decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim,
                dropout=config.dropout_rate,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
            self.temporal_decoder = nn.TransformerDecoder(
                decoder_layer=temporal_decoder_layer,
                num_layers=config.num_temporal_layers
            )

        # Le décodeur d'image final est utilisé pour générer chaque frame
        self.final_image_decoder = ImageDecoder(config.image_decoder_config)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialise les poids du modèle."""
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        latent: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Décode une représentation latente en vidéo.
        Si des cibles sont fournies, calcule aussi la perte de reconstruction.
        
        Args:
            latent: Représentation latente [batch_size, input_dim].
            targets: Vidéo cible [batch_size, C, T, H, W].
            
        Returns:
            Dictionnaire contenant :
            - 'output': Vidéo générée [batch_size, C, T, H, W].
            - 'loss': Perte de reconstruction (MSE) si `targets` est fourni.
        """
        generated_video = self.generate(latent)

        loss = torch.tensor(0.0, device=latent.device)
        if targets is not None:
            if generated_video.shape != targets.shape:
                # Adapter les dimensions spatiales et temporelles
                targets = F.interpolate(
                    targets, 
                    size=generated_video.shape[2:], # (T, H, W)
                    mode='trilinear', 
                    align_corners=False
                )
            loss = F.mse_loss(generated_video, targets)
            
        return {
            'output': generated_video,
            'loss': loss
        }

    @torch.no_grad()
    def generate(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Génère une vidéo à partir d'un vecteur latent (mode inférence).
        """
        batch_size = latent.shape[0]
        
        x = self.initial_projection(latent)
        x = x.view(batch_size, self.config.initial_channels, self.config.initial_time, self.config.initial_size, self.config.initial_size)
        
        for upsample_block in self.upsamples:
            x = upsample_block(x)
        
        x = self.output_conv(x)
        x = self.final_activation(x)
        
        # Redimensionnement final pour correspondre exactement à la sortie désirée
        if x.shape[2:] != (self.config.output_time, self.config.output_size, self.config.output_size):
             x = F.interpolate(
                x, 
                size=(self.config.output_time, self.config.output_size, self.config.output_size), 
                mode='trilinear', 
                align_corners=False
            )
            
        return x
