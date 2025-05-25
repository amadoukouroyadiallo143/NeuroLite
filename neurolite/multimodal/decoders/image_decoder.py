"""
Décodeur spécialisé pour la génération d'images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union, Tuple, Any


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


class ImageDecoder(nn.Module):
    """
    Décodeur pour la génération d'images à partir d'une représentation latente.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_channels: int = 3,
        initial_size: int = 7,
        initial_channels: int = 512,
        output_size: int = 224,
        num_upsamples: int = 5,
        use_residual: bool = True,
        dropout_rate: float = 0.1,
        final_activation: str = "tanh"
    ):
        """
        Initialise le décodeur d'images.
        
        Args:
            input_dim: Dimension d'entrée
            output_channels: Nombre de canaux de sortie (3 pour RGB)
            initial_size: Taille initiale de l'image spatiale
            initial_channels: Nombre de canaux initiaux
            output_size: Taille de sortie de l'image
            num_upsamples: Nombre d'étapes de suréchantillonnage
            use_residual: Si True, utilise des connexions résiduelles
            dropout_rate: Taux de dropout
            final_activation: Activation finale ("tanh", "sigmoid", "none")
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_channels = output_channels
        self.initial_size = initial_size
        self.initial_channels = initial_channels
        self.output_size = output_size
        
        # Vérifier si la taille de sortie est cohérente
        total_upsampling = 2 ** num_upsamples
        assert output_size % total_upsampling == 0, \
            f"output_size ({output_size}) must be divisible by total upsampling factor ({total_upsampling})"
        
        # Projection initiale du vecteur latent en feature map
        self.initial_projection = nn.Sequential(
            nn.Linear(input_dim, initial_channels * initial_size * initial_size),
            nn.Dropout(dropout_rate),
            nn.GELU()
        )
        
        # Blocs de suréchantillonnage
        self.upsamples = nn.ModuleList()
        
        current_channels = initial_channels
        for i in range(num_upsamples):
            # Réduire progressivement le nombre de canaux
            out_channels = current_channels // 2
            
            # Dernière couche : ajuster pour avoir une puissance de 2 près de la taille de sortie
            if i == num_upsamples - 1:
                # Calculer le facteur de mise à l'échelle nécessaire
                current_size = initial_size * (2 ** i)
                scale_factor = output_size // current_size
            else:
                scale_factor = 2
            
            self.upsamples.append(
                UpsampleBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    scale_factor=scale_factor,
                    use_residual=use_residual
                )
            )
            
            current_channels = out_channels
        
        # Couche de sortie
        self.output_conv = nn.Conv2d(
            in_channels=current_channels,
            out_channels=output_channels,
            kernel_size=3,
            padding=1
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
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Décode une représentation latente en image.
        
        Args:
            latent: Représentation latente [batch_size, input_dim]
            
        Returns:
            Image générée [batch_size, output_channels, output_size, output_size]
        """
        batch_size = latent.shape[0]
        
        # Projection initiale et reshaping
        x = self.initial_projection(latent)
        x = x.view(batch_size, self.initial_channels, self.initial_size, self.initial_size)
        
        # Blocs de suréchantillonnage
        for upsample_block in self.upsamples:
            x = upsample_block(x)
        
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
        Génère une image à partir d'une représentation latente.
        
        Args:
            latent: Représentation latente [batch_size, input_dim]
            temperature: Contrôle la variabilité des images générées
            noise_level: Niveau de bruit ajouté pendant la génération
            
        Returns:
            Image générée [batch_size, output_channels, output_size, output_size]
        """
        batch_size = latent.shape[0]
        
        # Ajouter du bruit au vecteur latent si nécessaire
        if noise_level > 0:
            noise = torch.randn_like(latent) * noise_level
            latent = latent + noise
        
        # Appliquer la température
        if temperature != 1.0:
            latent = latent / temperature
        
        # Générer l'image
        image = self.forward(latent)
        
        return image
