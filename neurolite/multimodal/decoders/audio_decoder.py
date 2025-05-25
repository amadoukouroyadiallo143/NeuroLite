"""
Décodeur spécialisé pour la génération d'audio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union, Tuple, Any


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


class AudioDecoder(nn.Module):
    """
    Décodeur pour la génération d'audio à partir d'une représentation latente.
    """
    
    def __init__(
        self,
        input_dim: int,
        initial_length: int = 16,
        initial_channels: int = 512,
        output_channels: int = 1,
        output_length: int = 16000,
        num_upsamples: int = 5,
        use_residual: bool = True,
        dropout_rate: float = 0.1,
        final_activation: str = "tanh"
    ):
        """
        Initialise le décodeur audio.
        
        Args:
            input_dim: Dimension d'entrée
            initial_length: Longueur initiale de la séquence audio
            initial_channels: Nombre de canaux initiaux
            output_channels: Nombre de canaux de sortie (généralement 1 pour audio mono)
            output_length: Longueur de sortie de l'audio
            num_upsamples: Nombre d'étapes de suréchantillonnage
            use_residual: Si True, utilise des connexions résiduelles
            dropout_rate: Taux de dropout
            final_activation: Activation finale ("tanh", "sigmoid", "none")
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.initial_length = initial_length
        self.initial_channels = initial_channels
        self.output_channels = output_channels
        self.output_length = output_length
        
        # Vérifier si la longueur de sortie est cohérente
        total_upsampling = 2 ** num_upsamples
        assert output_length % (total_upsampling * initial_length) == 0, \
            f"output_length ({output_length}) must be divisible by total upsampling factor * initial_length ({total_upsampling * initial_length})"
        
        # Calcul du facteur de mise à l'échelle final
        self.final_upsample_factor = output_length // (initial_length * (2 ** (num_upsamples - 1)))
        
        # Projection initiale du vecteur latent en séquence audio
        self.initial_projection = nn.Sequential(
            nn.Linear(input_dim, initial_channels * initial_length),
            nn.Dropout(dropout_rate),
            nn.GELU()
        )
        
        # Blocs de convolution montante
        self.upconv_blocks = nn.ModuleList()
        
        current_channels = initial_channels
        for i in range(num_upsamples):
            # Réduire progressivement le nombre de canaux
            out_channels = max(current_channels // 2, 32)
            
            # Dernier bloc : ajustement spécial
            if i == num_upsamples - 1:
                upsample_factor = self.final_upsample_factor
            else:
                upsample_factor = 2
            
            self.upconv_blocks.append(
                UpConvBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    upsample=True,
                    upsample_factor=upsample_factor,
                    use_residual=use_residual,
                    dropout_rate=dropout_rate
                )
            )
            
            current_channels = out_channels
        
        # Couche de sortie
        self.output_conv = nn.Conv1d(
            in_channels=current_channels,
            out_channels=output_channels,
            kernel_size=7,
            padding=3
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
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Décode une représentation latente en signal audio.
        
        Args:
            latent: Représentation latente [batch_size, input_dim]
            
        Returns:
            Signal audio généré [batch_size, output_channels, output_length]
        """
        batch_size = latent.shape[0]
        
        # Projection initiale et reshaping
        x = self.initial_projection(latent)
        x = x.view(batch_size, self.initial_channels, self.initial_length)
        
        # Blocs de convolution montante
        for upconv_block in self.upconv_blocks:
            x = upconv_block(x)
        
        # Couche de sortie
        x = self.output_conv(x)
        x = self.final_activation(x)
        
        return x
    
    def generate(
        self,
        latent: torch.Tensor,
        temperature: float = 1.0,
        noise_level: float = 0.0
    ) -> torch.Tensor:
        """
        Génère un signal audio à partir d'une représentation latente.
        
        Args:
            latent: Représentation latente [batch_size, input_dim]
            temperature: Contrôle la variabilité des signaux générés
            noise_level: Niveau de bruit ajouté pendant la génération
            
        Returns:
            Signal audio généré [batch_size, output_channels, output_length]
        """
        batch_size = latent.shape[0]
        
        # Ajouter du bruit au vecteur latent si nécessaire
        if noise_level > 0:
            noise = torch.randn_like(latent) * noise_level
            latent = latent + noise
        
        # Appliquer la température
        if temperature != 1.0:
            latent = latent / temperature
        
        # Générer le signal audio
        audio = self.forward(latent)
        
        return audio
