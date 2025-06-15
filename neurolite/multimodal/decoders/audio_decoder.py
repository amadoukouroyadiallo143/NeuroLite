"""
Décodeur spécialisé pour la génération d'audio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union, Tuple, Any
from dataclasses import dataclass

from neurolite.Configs.config import MMAudioDecoderConfig
from .base_decoder import BaseDecoder


class UpConvBlock(nn.Module):
    """
    Bloc de convolution montante pour le décodage audio.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        upsample: bool = True,
        upsample_factor: int = 2,
        use_residual: bool = True,
        dropout_rate: float = 0.1
    ):
        """
        Initialise un bloc de convolution montante.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            kernel_size: Taille du noyau de convolution
            stride: Pas de la convolution
            padding: Padding pour la convolution
            upsample: Si True, effectue un suréchantillonnage avant la convolution
            upsample_factor: Facteur de suréchantillonnage
            use_residual: Si True, ajoute une connexion résiduelle
            dropout_rate: Taux de dropout
        """
        super().__init__()
        
        self.use_residual = use_residual
        self.upsample = upsample
        
        # Suréchantillonnage
        if upsample:
            self.up = nn.Upsample(
                scale_factor=upsample_factor,
                mode="linear",
                align_corners=False
            )
        
        # Convolution
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        # Normalisation et activation
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Projection résiduelle si nécessaire
        if use_residual and (in_channels != out_channels or upsample):
            layers = []
            if upsample:
                layers.append(nn.Upsample(
                    scale_factor=upsample_factor,
                    mode="linear",
                    align_corners=False
                ))
            layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ))
            self.residual_proj = nn.Sequential(*layers)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applique le bloc de convolution montante.
        
        Args:
            x: Tenseur d'entrée [batch_size, in_channels, time]
            
        Returns:
            Tenseur traité [batch_size, out_channels, time']
        """
        identity = x
        
        # Suréchantillonnage si nécessaire
        if self.upsample:
            x = self.up(x)
        
        # Convolution et normalisation
        x = self.conv(x)
        x = self.norm(x)
        
        # Connexion résiduelle
        if self.use_residual:
            identity = self.residual_proj(identity)
            x = x + identity
        
        # Activation et dropout
        x = self.act(x)
        x = self.dropout(x)
        
        return x


class AudioDecoder(BaseDecoder):
    """
    Décode une représentation latente pour générer une forme d'onde audio.
    """
    def __init__(self, config: MMAudioDecoderConfig):
        """
        Initialise le décodeur audio.
        
        Args:
            config (MMAudioDecoderConfig): La configuration pour le décodeur audio.
        """
        super().__init__(config)
        self.config = config
        
        # Projection initiale du vecteur latent en séquence audio
        self.initial_projection = nn.Sequential(
            nn.Linear(config.input_dim, config.initial_channels * config.initial_length),
            nn.Dropout(config.dropout_rate),
            nn.GELU()
        )
        
        # Blocs de convolution montante
        self.upconv_blocks = nn.ModuleList()
        current_channels = config.initial_channels
        
        for i in range(config.num_upsamples):
            out_channels = max(current_channels // 2, 32)
            
            self.upconv_blocks.append(
                UpConvBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=5,
                    padding=2,
                    upsample_factor=2,
                    use_residual=config.use_residual,
                    dropout_rate=config.dropout_rate
                )
            )
            current_channels = out_channels
        
        # Couche de sortie
        self.output_conv = nn.Conv1d(
            in_channels=current_channels,
            out_channels=config.output_channels,
            kernel_size=7,
            padding=3
        )
        
        # Activation finale
        if config.final_activation == "tanh":
            self.final_activation = nn.Tanh()
        elif config.final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()
            
        self.feature_upsampler = nn.Linear(config.hidden_dim, config.n_mels * 2)
            
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialise les poids du modèle."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
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
        Décode une représentation latente en signal audio.
        Si des cibles sont fournies, calcule aussi la perte de reconstruction.

        Args:
            latent: Représentation latente [batch_size, input_dim].
            targets: Signal audio cible [batch_size, channels, length].

        Returns:
            Dictionnaire contenant :
            - 'output': Signal audio généré [batch_size, C, L].
            - 'loss': Perte de reconstruction (L1) si `targets` est fourni.
        """
        generated_audio = self.generate(latent)
        
        # Calcul de la perte si les cibles sont fournies
        loss = torch.tensor(0.0, device=latent.device)
        if targets is not None:
            # Assurer la même longueur pour la perte
            if generated_audio.shape[-1] != targets.shape[-1]:
                targets = F.interpolate(targets, size=generated_audio.shape[-1], mode='linear', align_corners=False)
            loss = F.l1_loss(generated_audio, targets)
            
        return {
            'output': generated_audio,
            'loss': loss
        }

    @torch.no_grad()
    def generate(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Génère un signal audio à partir d'un vecteur latent (mode inférence).
        """
        batch_size = latent.shape[0]
        
        x = self.initial_projection(latent)
        x = x.view(batch_size, self.config.initial_channels, self.config.initial_length)
        
        for upconv_block in self.upconv_blocks:
            x = upconv_block(x)
        
        # Adapter la longueur finale
        if x.shape[-1] != self.config.output_length:
             x = F.interpolate(x, size=(self.config.output_length,), mode='linear', align_corners=False)

        # Couche de sortie et activation
        x = self.output_conv(x)
        x = self.final_activation(x)
        
        return x
