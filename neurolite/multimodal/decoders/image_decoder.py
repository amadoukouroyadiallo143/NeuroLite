"""
Décodeur spécialisé pour la génération d'images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union, Tuple, Any
from dataclasses import dataclass

from neurolite.Configs.config import MMImageDecoderConfig
from .base_decoder import BaseDecoder


class UpsampleBlock(nn.Module):
    """
    Bloc de suréchantillonnage pour le décodage d'images.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        use_residual: bool = True,
        norm_type: str = "batch",
        activation: str = "gelu"
    ):
        """
        Initialise un bloc de suréchantillonnage.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            scale_factor: Facteur de mise à l'échelle
            use_residual: Si True, ajoute une connexion résiduelle
            norm_type: Type de normalisation ("batch", "layer", "instance", "none")
            activation: Fonction d'activation ("relu", "gelu", "silu", "none")
        """
        super().__init__()
        
        self.use_residual = use_residual
        
        # Couche de suréchantillonnage
        self.upsample = nn.Upsample(
            scale_factor=scale_factor,
            mode="nearest"
        )
        
        # Convolution
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=norm_type == "none"
        )
        
        # Normalisation
        if norm_type == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == "layer":
            self.norm = nn.GroupNorm(1, out_channels)
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm2d(out_channels)
        else:  # "none"
            self.norm = nn.Identity()
        
        # Activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU(inplace=True)
        else:  # "none"
            self.activation = nn.Identity()
        
        # Projection résiduelle si nécessaire
        if use_residual and (in_channels != out_channels or scale_factor != 1):
            self.residual_proj = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode="nearest"),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Suréchantillonne l'entrée.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur suréchantillonné [B, C_out, H*scale, W*scale]
        """
        identity = x
        
        # Suréchantillonnage + convolution
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        
        # Ajouter la connexion résiduelle si nécessaire
        if self.use_residual:
            x = x + self.residual_proj(identity)
        
        # Activation
        x = self.activation(x)
        
        return x


class ImageDecoder(BaseDecoder):
    """
    Décode une représentation latente pour générer une image.
    """
    def __init__(self, config: MMImageDecoderConfig):
        """
        Initialise le décodeur d'image.
        
        Args:
            config (MMImageDecoderConfig): La configuration pour le décodeur d'image.
        """
        super().__init__(config)
        self.config = config
        
        # Vérifier si la taille de sortie est cohérente
        total_upsampling = 2 ** config.num_upsamples
        if config.target_image_size % (config.initial_size * total_upsampling) != 0:
            raise ValueError(
                f"La taille de sortie de l'image ({config.target_image_size}) n'est pas atteignable "
                f"avec initial_size={config.initial_size} et num_upsamples={config.num_upsamples}."
            )
        
        # Projection initiale du vecteur latent en feature map
        self.initial_projection = nn.Sequential(
            nn.Linear(config.input_dim, config.initial_channels * config.initial_size * config.initial_size),
            nn.Dropout(config.dropout_rate),
            nn.GELU()
        )
        
        # Blocs de suréchantillonnage
        self.upsamples = nn.ModuleList()
        
        current_channels = config.initial_channels
        for i in range(config.num_upsamples):
            out_channels = max(current_channels // 2, 32)
            
            self.upsamples.append(
                UpsampleBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    scale_factor=2, # Chaque bloc double la résolution spatiale
                    use_residual=config.use_residual
                )
            )
            current_channels = out_channels
        
        # Couche de sortie
        self.output_conv = nn.Conv2d(
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
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialise les poids du modèle."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        latent: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Décode une représentation latente en image.
        Si des cibles sont fournies, calcule aussi la perte de reconstruction.

        Args:
            latent: Représentation latente [batch_size, input_dim].
            targets: Image cible [batch_size, channels, height, width].

        Returns:
            Dictionnaire contenant :
            - 'output': Image générée [batch_size, C, H, W].
            - 'loss': Perte de reconstruction (MSE) si `targets` est fourni.
        """
        generated_image = self.generate(latent)
        
        # Calcul de la perte si les cibles sont fournies
        loss = torch.tensor(0.0, device=latent.device)
        if targets is not None:
            # S'assurer que les tailles correspondent pour le calcul de la perte
            if generated_image.shape != targets.shape:
                 targets = F.interpolate(targets, size=generated_image.shape[2:], mode='bilinear', align_corners=False)
            loss = F.mse_loss(generated_image, targets)
            
        return {
            'output': generated_image,
            'loss': loss
        }

    @torch.no_grad()
    def generate(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Génère une image à partir d'un vecteur latent (mode inférence).
        """
        batch_size = latent.shape[0]
        
        # Projection initiale et reshaping
        x = self.initial_projection(latent)
        x = x.view(batch_size, self.config.initial_channels, self.config.initial_size, self.config.initial_size)
        
        # Blocs de suréchantillonnage
        for upsample_block in self.upsamples:
            x = upsample_block(x)
        
        # Couche de sortie
        x = self.output_conv(x)
        x = self.final_activation(x)
        
        return x
