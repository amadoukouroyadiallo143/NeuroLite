"""
Décodeur spécialisé pour la génération de vidéos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union, Tuple, Any

from .image_decoder import ImageDecoder, UpsampleBlock


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


class VideoDecoder(nn.Module):
    """
    Décodeur pour la génération de vidéos à partir d'une représentation latente.
    """
    
    def __init__(
        self,
        input_dim: int,
        initial_time: int = 2,
        initial_size: int = 7,
        initial_channels: int = 512,
        output_channels: int = 3,
        output_time: int = 16,
        output_size: int = 224,
        num_temporal_upsamples: int = 3,
        num_spatial_upsamples: int = 5,
        use_image_decoder: bool = False,
        use_residual: bool = True,
        dropout_rate: float = 0.1,
        final_activation: str = "tanh"
    ):
        """
        Initialise le décodeur vidéo.
        
        Args:
            input_dim: Dimension d'entrée
            initial_time: Longueur temporelle initiale
            initial_size: Taille spatiale initiale
            initial_channels: Nombre de canaux initiaux
            output_channels: Nombre de canaux de sortie (3 pour RGB)
            output_time: Longueur temporelle de sortie
            output_size: Taille spatiale de sortie
            num_temporal_upsamples: Nombre d'étapes de suréchantillonnage temporel
            num_spatial_upsamples: Nombre d'étapes de suréchantillonnage spatial
            use_image_decoder: Si True, utilise un décodeur d'image pour chaque trame
            use_residual: Si True, utilise des connexions résiduelles
            dropout_rate: Taux de dropout
            final_activation: Activation finale ("tanh", "sigmoid", "none")
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.initial_time = initial_time
        self.initial_size = initial_size
        self.initial_channels = initial_channels
        self.output_channels = output_channels
        self.output_time = output_time
        self.output_size = output_size
        self.use_image_decoder = use_image_decoder
        
        # Vérifier la cohérence des dimensions
        temporal_upsampling = 2 ** num_temporal_upsamples
        spatial_upsampling = 2 ** num_spatial_upsamples
        
        assert output_time % (temporal_upsampling * initial_time) == 0, \
            f"output_time ({output_time}) must be divisible by total temporal upsampling factor ({temporal_upsampling * initial_time})"
        
        assert output_size % (spatial_upsampling * initial_size) == 0, \
            f"output_size ({output_size}) must be divisible by total spatial upsampling factor ({spatial_upsampling * initial_size})"
        
        # Facteurs de mise à l'échelle finaux
        self.final_temporal_factor = output_time // (initial_time * (2 ** (num_temporal_upsamples - 1)))
        self.final_spatial_factor = output_size // (initial_size * (2 ** (num_spatial_upsamples - 1)))
        
        # Projection initiale du vecteur latent
        self.initial_projection = nn.Sequential(
            nn.Linear(input_dim, initial_channels * initial_time * initial_size * initial_size),
            nn.Dropout(dropout_rate),
            nn.GELU()
        )
        
        if use_image_decoder:
            # Utiliser un décodeur d'image pour chaque trame
            self.image_decoder = ImageDecoder(
                input_dim=input_dim // initial_time,
                output_channels=output_channels,
                initial_size=initial_size,
                initial_channels=initial_channels,
                output_size=output_size,
                num_upsamples=num_spatial_upsamples,
                use_residual=use_residual,
                dropout_rate=dropout_rate,
                final_activation=final_activation
            )
            
            # Projection temporelle
            self.temporal_projection = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Linear(input_dim, output_time * (input_dim // initial_time))
            )
        else:
            # Blocs de suréchantillonnage combiné spatio-temporel
            self.upconv_blocks = nn.ModuleList()
            
            current_channels = initial_channels
            
            # D'abord, effectuer le suréchantillonnage temporel
            for i in range(num_temporal_upsamples):
                # Réduire progressivement le nombre de canaux
                out_channels = max(current_channels // 2, 64)
                
                # Dernier bloc temporel : ajustement spécial
                if i == num_temporal_upsamples - 1:
                    temporal_factor = self.final_temporal_factor
                else:
                    temporal_factor = 2
                
                self.upconv_blocks.append(
                    TemporalUpsampleBlock(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        temporal_kernel_size=3,
                        spatial_kernel_size=3,
                        temporal_upsample=True,
                        spatial_upsample=False,
                        upsample_factor=temporal_factor,
                        use_residual=use_residual,
                        dropout_rate=dropout_rate
                    )
                )
                
                current_channels = out_channels
            
            # Ensuite, effectuer le suréchantillonnage spatial
            for i in range(num_spatial_upsamples):
                # Réduire progressivement le nombre de canaux
                out_channels = max(current_channels // 2, 32)
                
                # Dernier bloc spatial : ajustement spécial
                if i == num_spatial_upsamples - 1:
                    spatial_factor = self.final_spatial_factor
                else:
                    spatial_factor = 2
                
                self.upconv_blocks.append(
                    TemporalUpsampleBlock(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        temporal_kernel_size=3,
                        spatial_kernel_size=3,
                        temporal_upsample=False,
                        spatial_upsample=True,
                        upsample_factor=spatial_factor,
                        use_residual=use_residual,
                        dropout_rate=dropout_rate
                    )
                )
                
                current_channels = out_channels
            
            # Couche de sortie
            self.output_conv = nn.Conv3d(
                in_channels=current_channels,
                out_channels=output_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1)
            )
            
            # Activation finale
            if final_activation == "tanh":
                self.final_activation = nn.Tanh()
            elif final_activation == "sigmoid":
                self.final_activation = nn.Sigmoid()
            else:  # "none"
                self.final_activation = nn.Identity()
        
        # Initialisation des poids
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialise les poids du modèle."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Décode une représentation latente en vidéo.
        
        Args:
            latent: Représentation latente [batch_size, input_dim]
            
        Returns:
            Vidéo générée [batch_size, output_channels, output_time, output_size, output_size]
        """
        batch_size = latent.shape[0]
        
        if self.use_image_decoder:
            # Projeter le vecteur latent en multiples vecteurs latents pour chaque trame
            temporal_latents = self.temporal_projection(latent)
            temporal_latents = temporal_latents.view(batch_size, self.output_time, -1)
            
            # Générer chaque trame indépendamment
            frames = []
            for t in range(self.output_time):
                frame_latent = temporal_latents[:, t]
                frame = self.image_decoder(frame_latent)
                frames.append(frame)
            
            # Concaténer les trames
            video = torch.stack(frames, dim=2)  # [B, C, T, H, W]
        else:
            # Projection initiale et reshaping
            x = self.initial_projection(latent)
            x = x.view(
                batch_size,
                self.initial_channels,
                self.initial_time,
                self.initial_size,
                self.initial_size
            )
            
            # Blocs de suréchantillonnage
            for upconv_block in self.upconv_blocks:
                x = upconv_block(x)
            
            # Couche de sortie
            x = self.output_conv(x)
            video = self.final_activation(x)
        
        return video
    
    def generate(
        self,
        latent: torch.Tensor,
        temperature: float = 1.0,
        noise_level: float = 0.0
    ) -> torch.Tensor:
        """
        Génère une vidéo à partir d'une représentation latente.
        
        Args:
            latent: Représentation latente [batch_size, input_dim]
            temperature: Contrôle la variabilité des vidéos générées
            noise_level: Niveau de bruit ajouté pendant la génération
            
        Returns:
            Vidéo générée [batch_size, output_channels, output_time, output_size, output_size]
        """
        batch_size = latent.shape[0]
        
        # Ajouter du bruit au vecteur latent si nécessaire
        if noise_level > 0:
            noise = torch.randn_like(latent) * noise_level
            latent = latent + noise
        
        # Appliquer la température
        if temperature != 1.0:
            latent = latent / temperature
        
        # Générer la vidéo
        video = self.forward(latent)
        
        return video
