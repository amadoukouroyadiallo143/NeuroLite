"""
Module MLP-Mixer pour NeuroLite.
Implémente différentes variantes de MLP-Mixer pour le traitement efficace de séquences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Callable, Union, List, Dict


class MLPBlock(nn.Module):
    """
    Bloc MLP de base avec projections, activation, dropout et normalisation.
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout_rate: float = 0.0,
        activation: Union[str, Callable] = "gelu"
    ):
        super().__init__()
        
        # Sélection de la fonction d'activation
        if isinstance(activation, str):
            self.activation = {
                "gelu": F.gelu,
                "relu": F.relu,
                "silu": F.silu,
            }[activation.lower()]
        else:
            self.activation = activation
            
        # Couches du MLP
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passage avant dans le bloc MLP"""
        # Projection montante
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Projection descendante
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x


class MixerLayer(nn.Module):
    """
    Couche MLP-Mixer complète avec token-mixing et channel-mixing.
    Basée sur l'architecture MLP-Mixer (Tolstikhin et al., 2021).
    """
    
    def __init__(
        self,
        dim: int,
        seq_len: int,
        token_mixing_hidden_dim: int,
        channel_mixing_hidden_dim: int,
        dropout_rate: float = 0.0,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-6
    ):
        super().__init__()
        
        # Normalisation
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        
        # Paramètres pour le token-mixing dynamique
        self.max_seq_len = seq_len
        self.token_mixing_hidden_dim = token_mixing_hidden_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        # Initial token_mix MLP (will be replaced if seq_len changes in forward pass)
        # This ensures the submodule exists for state_dict loading.
        # The dim for token_mix MLP is seq_len because it mixes across tokens for each channel.
        # The hidden_dim calculation matches what _create_token_mix would do for max_seq_len.
        _initial_token_mix_mlp_input_dim = self.max_seq_len
        _initial_token_mix_mlp_hidden_dim = min(self.token_mixing_hidden_dim, self.max_seq_len * 2)
        self.token_mix = MLPBlock(
            dim=_initial_token_mix_mlp_input_dim, 
            hidden_dim=_initial_token_mix_mlp_hidden_dim,
            dropout_rate=self.dropout_rate,
            activation=self.activation
        )
        
        # MLP pour le channel-mixing (mélange entre features)
        self.channel_mix = MLPBlock(
            dim=dim,
            hidden_dim=channel_mixing_hidden_dim,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant dans la couche MLP-Mixer
        
        Args:
            x: Tensor d'entrée [batch_size, seq_len, dim]
            
        Returns:
            Tensor transformé [batch_size, seq_len, dim]
        """
        batch_size, current_seq_len, model_dim = x.shape
        
        # Token-mixing
        residual_token_mix = x
        x_norm_token_mix = self.norm1(x) # shape: [b, current_seq_len, model_dim]
        
        # Transpose for token mixing: MLP acts on features of size current_seq_len (for each of model_dim channels)
        x_transposed = rearrange(x_norm_token_mix, 'b s d -> b d s') # shape: [b, model_dim, current_seq_len]
        
        # Pad if current_seq_len is less than self.max_seq_len (which token_mix expects)
        if current_seq_len < self.max_seq_len:
            padding_size = self.max_seq_len - current_seq_len
            # Pad on the right of the last dimension (sequence dimension for x_transposed)
            x_padded = F.pad(x_transposed, (0, padding_size))
        elif current_seq_len == self.max_seq_len:
            x_padded = x_transposed
        else: # current_seq_len > self.max_seq_len
            # Truncate if input sequence is longer than what the token_mix MLP is configured for.
            # This loses information but makes the model runnable. A better solution might be an error or sliding window.
            x_padded = x_transposed[:, :, :self.max_seq_len]
            
        x_mixed_padded = self.token_mix(x_padded) # token_mix output: [b, model_dim, self.max_seq_len]
        
        # Truncate if padding was applied or if original input was truncated
        if current_seq_len < self.max_seq_len:
            x_mixed = x_mixed_padded[:, :, :current_seq_len] # Truncate back to original current_seq_len
        else: # current_seq_len >= self.max_seq_len (covers equality and the truncation case)
            x_mixed = x_mixed_padded # Output is already at self.max_seq_len (either due to no padding or input truncation)

        x_untransposed = rearrange(x_mixed, 'b d s -> b s d') # shape: [b, current_seq_len or self.max_seq_len, model_dim]
        x = x_untransposed + residual_token_mix
        
        # Channel-mixing
        residual_channel_mix = x
        x_norm_channel_mix = self.norm2(x)
        x = self.channel_mix(x_norm_channel_mix)
        x = x + residual_channel_mix
        
        return x


class HyperMixer(nn.Module):
    """
    Implémentation du HyperMixer (Mai et al., 2023).
    Utilise un petit hyper-réseau pour générer dynamiquement les paramètres
    du token-mixing en fonction de la longueur d'entrée.
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        token_mixing_hidden_dim: int,
        channel_mixing_hidden_dim: int,
        dropout_rate: float = 0.0,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-6,
        bottleneck_dim: int = 16
    ):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.token_mixing_hidden_dim = token_mixing_hidden_dim
        
        # Normalisation
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        
        # Hyper-réseau pour générer les paramètres de token-mixing
        self.hyper_net = nn.Sequential(
            nn.Linear(1, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, 2 * max_seq_len * token_mixing_hidden_dim + 
                      token_mixing_hidden_dim + max_seq_len)
        )
        
        # MLP pour le channel-mixing (mélange entre features)
        self.channel_mix = MLPBlock(
            dim=dim,
            hidden_dim=channel_mixing_hidden_dim,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    def _generate_token_mixing_weights(self, seq_len: int) -> Dict[str, torch.Tensor]:
        """
        Génère les poids pour le token-mixing à partir de l'hyper-réseau.
        """
        # Normaliser la longueur de séquence pour l'entrée du hyper-réseau
        norm_seq_len = torch.tensor([[seq_len / self.max_seq_len]], device=self.norm1.weight.device)
        
        # Obtenir les paramètres générés
        params = self.hyper_net(norm_seq_len)
        
        # Nombre total de paramètres
        total_params = 2 * seq_len * self.token_mixing_hidden_dim + self.token_mixing_hidden_dim + seq_len
        
        # Si la longueur est plus petite que max_seq_len, tronquer les paramètres
        params = params[:, :total_params]
        
        # Restructurer en matrices et biais
        w1_size = seq_len * self.token_mixing_hidden_dim
        w2_size = self.token_mixing_hidden_dim * seq_len
        b1_size = self.token_mixing_hidden_dim
        b2_size = seq_len
        
        # Découper le vecteur de paramètres
        w1 = params[:, :w1_size].reshape(1, seq_len, self.token_mixing_hidden_dim)
        w2 = params[:, w1_size:w1_size+w2_size].reshape(1, self.token_mixing_hidden_dim, seq_len)
        b1 = params[:, w1_size+w2_size:w1_size+w2_size+b1_size].reshape(1, self.token_mixing_hidden_dim)
        b2 = params[:, w1_size+w2_size+b1_size:].reshape(1, seq_len)
        
        return {
            "w1": w1.squeeze(0),
            "w2": w2.squeeze(0),
            "b1": b1.squeeze(0),
            "b2": b2.squeeze(0)
        }
        
    def _token_mixing(self, x: torch.Tensor, weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Applique le token-mixing avec les poids générés dynamiquement.
        """
        # x: [batch_size, dim, seq_len]
        batch_size, dim, seq_len = x.shape
        
        # Première couche: x @ w1 + b1
        # [batch_size, dim, seq_len] @ [seq_len, hidden_dim] + [hidden_dim]
        h = torch.bmm(x, weights["w1"].unsqueeze(0).expand(batch_size, -1, -1))
        h = h + weights["b1"].unsqueeze(0).unsqueeze(1).expand(batch_size, dim, -1)
        h = F.gelu(h)
        h = self.dropout(h)
        
        # Deuxième couche: h @ w2 + b2
        # [batch_size, dim, hidden_dim] @ [hidden_dim, seq_len] + [seq_len]
        out = torch.bmm(h, weights["w2"].unsqueeze(0).expand(batch_size, -1, -1))
        out = out + weights["b2"].unsqueeze(0).unsqueeze(1).expand(batch_size, dim, -1)
        out = self.dropout(out)
        
        return out
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant dans le HyperMixer
        
        Args:
            x: Tensor d'entrée [batch_size, seq_len, dim]
            
        Returns:
            Tensor transformé [batch_size, seq_len, dim]
        """
        batch_size, seq_len, dim = x.shape
        
        # Token-mixing avec poids dynamiques
        residual = x
        x = self.norm1(x)
        x_transposed = rearrange(x, 'b s d -> b d s')
        
        # Générer les poids pour cette longueur spécifique
        token_mix_weights = self._generate_token_mixing_weights(seq_len)
        
        # Appliquer le token-mixing dynamique
        x_mixed = self._token_mixing(x_transposed, token_mix_weights)
        x = rearrange(x_mixed, 'b d s -> b s d')
        x = x + residual
        
        # Channel-mixing (identique à MLP-Mixer standard)
        residual = x
        x = self.norm2(x)
        x = self.channel_mix(x)
        x = x + residual
        
        return x


class FNetLayer(nn.Module):
    """
    Implémentation d'une couche FNet (Lee et al., 2022).
    Remplace l'attention par des transformées de Fourier discrètes.
    """
    
    def __init__(
        self,
        dim: int,
        channel_mixing_hidden_dim: int,
        dropout_rate: float = 0.0,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-6
    ):
        super().__init__()
        
        # Normalisation
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        
        # MLP pour le channel-mixing
        self.channel_mix = MLPBlock(
            dim=dim,
            hidden_dim=channel_mixing_hidden_dim,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
    def _fft_mixing(self, x: torch.Tensor) -> torch.Tensor:
        """Applique deux FFTs successives pour le mélange"""
        # FFT le long de la dimension des tokens (séquence)
        x = torch.fft.fft(x, dim=1).real
        # FFT le long de la dimension des features
        x = torch.fft.fft(x, dim=2).real
        return x
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant dans la couche FNet
        
        Args:
            x: Tensor d'entrée [batch_size, seq_len, dim]
            
        Returns:
            Tensor transformé [batch_size, seq_len, dim]
        """
        # FFT mixing
        residual = x
        x = self.norm1(x)
        x = self._fft_mixing(x)
        x = x + residual
        
        # Channel-mixing (comme dans MLP-Mixer)
        residual = x
        x = self.norm2(x)
        x = self.channel_mix(x)
        x = x + residual
        
        return x
