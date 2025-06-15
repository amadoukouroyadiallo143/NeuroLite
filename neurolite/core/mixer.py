"""
Module MLP-Mixer pour NeuroLite.
Implémente différentes variantes de MLP-Mixer pour le traitement efficace de séquences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Callable, Union, List, Dict
import warnings


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
        batch_size, current_seq_len, _ = x.shape
        
        if current_seq_len > self.max_seq_len:
            warnings.warn(
                f"Input sequence length ({current_seq_len}) to MixerLayer is greater than max_seq_len "
                f"({self.max_seq_len}). MixerLayer is not designed to extrapolate weights beyond max_seq_len. "
                f"Behavior may be unpredictable or error-prone.",
                RuntimeWarning
            )
            # Optionnel: tronquer current_seq_len pour éviter des erreurs d'indexation plus loin,
            # ou laisser l'erreur se produire si c'est préférable.
            # current_seq_len = self.max_seq_len # Décommenter pour forcer la troncature
        
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
            warnings.warn(
                f"Input sequence length ({current_seq_len}) to MixerLayer is greater than max_seq_len "
                f"({self.max_seq_len}). Input will be truncated, leading to information loss.",
                RuntimeWarning
            )
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
        
        # Ensure residual_token_mix matches x_untransposed's sequence length
        if residual_token_mix.shape[1] != x_untransposed.shape[1]:
            residual_token_mix = residual_token_mix[:, :x_untransposed.shape[1], :]
            
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
    du token-mixing. L'hyper-réseau génère des poids dimensionnés pour `max_seq_len`.
    Si `current_seq_len < max_seq_len`, des sous-ensembles de ces poids sont utilisés.
    Si `current_seq_len > max_seq_len`, un avertissement est émis car le modèle n'est pas
    conçu pour extrapoler au-delà de `max_seq_len`.
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
        if isinstance(activation, str):
            self.activation = {
                "gelu": F.gelu,
                "relu": F.relu,
                "silu": F.silu,
            }.get(activation.lower(), F.gelu) # Default to gelu if string not found
        else:
            self.activation = activation
        self.dropout = nn.Dropout(dropout_rate) # Dropout layer for dynamic MLP
        self.dropout_rate = dropout_rate # Retain for reference if needed elsewhere, though self.dropout is primary
        self.layer_norm_eps = layer_norm_eps # Store for dynamic MLP
        
        # Normalisation
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        
        # Hyper-réseau pour générer les paramètres de token-mixing.
        # Il génère assez de paramètres pour un token_mix_mlp opérant sur max_seq_len.
        # Pour fc1: poids (max_seq_len * token_mixing_hidden_dim) + biais (token_mixing_hidden_dim)
        # Pour fc2: poids (token_mixing_hidden_dim * max_seq_len) + biais (max_seq_len)
        # Total: 2 * max_seq_len * token_mixing_hidden_dim + token_mixing_hidden_dim + max_seq_len
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
        
        # Dropout layer for use in forward if needed (though MLPBlock has its own)
        # self.dropout = nn.Dropout(dropout_rate) # This was likely for the old _token_mixing method
        
    def _generate_token_mix_mlp_params(self, target_seq_len: int) -> Dict[str, torch.Tensor]:
        """
        Génère les poids et biais pour le MLP de token-mixing via l'hyper-réseau.
        L'hyper-réseau est conçu pour générer des paramètres pour `self.max_seq_len`.
        Cette fonction extrait ces paramètres.
        """
        # L'entrée du hyper_net est normalisée, mais il génère toujours des poids pour max_seq_len.
        # On utilise une valeur fixe (ex: 1.0) pour indiquer qu'on veut les poids pour max_seq_len.
        norm_input_for_hypernet = torch.tensor([[1.0]], device=self.hyper_net[0].weight.device)
        
        # Obtenir le vecteur de paramètres plats généré par l'hyper-réseau
        # Ce vecteur contient les poids pour un MLP opérant sur max_seq_len
        flat_params = self.hyper_net(norm_input_for_hypernet).squeeze(0)

        # Découper flat_params pour obtenir w1, b1, w2, b2 pour max_seq_len
        # Tailles attendues pour max_seq_len:
        # w1: (token_mixing_hidden_dim, max_seq_len)
        # b1: (token_mixing_hidden_dim)
        # w2: (max_seq_len, token_mixing_hidden_dim)
        # b2: (max_seq_len)

        idx = 0
        w1_size = self.token_mixing_hidden_dim * self.max_seq_len
        w1 = flat_params[idx : idx + w1_size].reshape(self.token_mixing_hidden_dim, self.max_seq_len)
        idx += w1_size

        b1_size = self.token_mixing_hidden_dim
        b1 = flat_params[idx : idx + b1_size]
        idx += b1_size

        w2_size = self.max_seq_len * self.token_mixing_hidden_dim
        w2 = flat_params[idx : idx + w2_size].reshape(self.max_seq_len, self.token_mixing_hidden_dim)
        idx += w2_size

        b2_size = self.max_seq_len
        b2 = flat_params[idx : idx + b2_size]
        
        return {
            "w1_max": w1, # Poids pour max_seq_len (token_mixing_hidden_dim, max_seq_len)
            "b1_max": b1, # Biais (token_mixing_hidden_dim)
            "w2_max": w2, # Poids pour max_seq_len (max_seq_len, token_mixing_hidden_dim)
            "b2_max": b2  # Biais pour max_seq_len (max_seq_len)
        }
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant dans le HyperMixer
        
        Args:
            x: Tensor d'entrée [batch_size, seq_len, dim]
            
        Returns:
            Tensor transformé [batch_size, seq_len, dim]
        """
        batch_size, current_seq_len, dim = x.shape
        if current_seq_len > self.max_seq_len:
            warnings.warn(
                f"HyperMixer: Input sequence length ({current_seq_len}) > max_seq_len ({self.max_seq_len}). Truncating.",
                RuntimeWarning
            )
            x = x[:, :self.max_seq_len, :]
            current_seq_len = self.max_seq_len
        
        # Token-mixing avec poids dynamiques
        # Token-mixing
        residual_token_mix = x 
        x_norm_token_mix = self.norm1(x) 
        x_transposed = rearrange(x_norm_token_mix, 'b s d -> b d s')
        
        all_params_max = self._generate_token_mix_mlp_params(self.max_seq_len)

        w1 = all_params_max["w1_max"][:, :current_seq_len] 
        b1 = all_params_max["b1_max"] 
        w2 = all_params_max["w2_max"][:current_seq_len, :] 
        b2 = all_params_max["b2_max"][:current_seq_len] 

        y = F.linear(x_transposed, w1, b1)
        y = self.activation(y)
        y = self.dropout(y)
        y = F.linear(y, w2, b2)
        y = self.dropout(y)
        
        x_untransposed = rearrange(y, 'b d s -> b s d')
        x = x_untransposed + residual_token_mix
        
        # Channel-mixing
        residual_channel_mix = x 
        x_norm_channel_mix = self.norm2(x) 
        x = self.channel_mix(x_norm_channel_mix) 
        x = x + residual_channel_mix
        
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


if __name__ == "__main__":
    print("\n--- Testing HyperMixer ---")
    
    # HyperMixer Parameters
    h_dim = 64
    h_max_seq_len = 32
    h_token_mix_hidden = 128
    h_channel_mix_hidden = 256
    h_dropout = 0.0 # Keep dropout 0 for deterministic shape tests
    
    hyper_mixer_layer = HyperMixer(
        dim=h_dim,
        max_seq_len=h_max_seq_len,
        token_mixing_hidden_dim=h_token_mix_hidden,
        channel_mixing_hidden_dim=h_channel_mix_hidden,
        dropout_rate=h_dropout,
        activation="gelu"
    )
    hyper_mixer_layer.eval() # Set to eval mode for testing (disables dropout if > 0)

    batch_size = 2
    
    # Test case 1: current_seq_len < max_seq_len
    current_seq_len_1 = 16
    print(f"\nTest 1: HyperMixer with current_seq_len ({current_seq_len_1}) < max_seq_len ({h_max_seq_len})")
    dummy_input_1 = torch.randn(batch_size, current_seq_len_1, h_dim)
    output_1 = hyper_mixer_layer(dummy_input_1)
    print(f"Input shape: {dummy_input_1.shape}")
    print(f"Output shape: {output_1.shape}")
    assert output_1.shape == dummy_input_1.shape, f"Test 1 failed: Output shape {output_1.shape} != Input shape {dummy_input_1.shape}"
    print("Test 1 Passed.")

    # Test case 2: current_seq_len == max_seq_len
    current_seq_len_2 = h_max_seq_len
    print(f"\nTest 2: HyperMixer with current_seq_len ({current_seq_len_2}) == max_seq_len ({h_max_seq_len})")
    dummy_input_2 = torch.randn(batch_size, current_seq_len_2, h_dim)
    output_2 = hyper_mixer_layer(dummy_input_2)
    print(f"Input shape: {dummy_input_2.shape}")
    print(f"Output shape: {output_2.shape}")
    assert output_2.shape == dummy_input_2.shape, f"Test 2 failed: Output shape {output_2.shape} != Input shape {dummy_input_2.shape}"
    print("Test 2 Passed.")

    # Test case 3: current_seq_len > max_seq_len (expects warning and truncation)
    current_seq_len_3 = 48
    expected_output_seq_len_3 = h_max_seq_len # Due to truncation
    print(f"\nTest 3: HyperMixer with current_seq_len ({current_seq_len_3}) > max_seq_len ({h_max_seq_len})")
    dummy_input_3 = torch.randn(batch_size, current_seq_len_3, h_dim)
    
    print("Expecting a RuntimeWarning for sequence length truncation...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always") # Capture all warnings
        output_3 = hyper_mixer_layer(dummy_input_3)
        
        assert len(w) > 0, "Test 3 failed: No warning was raised for sequence truncation."
        assert issubclass(w[-1].category, RuntimeWarning), "Test 3 failed: Warning was not a RuntimeWarning."
        print(f"Caught warning: {w[-1].message}")
        
    print(f"Input shape: {dummy_input_3.shape}")
    print(f"Output shape: {output_3.shape}")
    expected_shape_3 = (batch_size, expected_output_seq_len_3, h_dim)
    assert output_3.shape == expected_shape_3, f"Test 3 failed: Output shape {output_3.shape} != Expected shape {expected_shape_3}"
    print("Test 3 Passed (Warning caught and output shape is correct after truncation).")

    print("\n--- HyperMixer tests completed ---")

    print("\n--- Testing MixerLayer (simple) ---")
    m_dim = 64
    m_seq_len = 32 # This is max_seq_len for MixerLayer's token_mix MLP
    m_token_mix_hidden = 512 # Typically larger for fixed MLP
    m_channel_mix_hidden = 256
    
    mixer_layer_instance = MixerLayer(
        dim=m_dim,
        seq_len=m_seq_len, # This is the 'max_seq_len' the token_mix MLP is built for
        token_mixing_hidden_dim=m_token_mix_hidden,
        channel_mixing_hidden_dim=m_channel_mix_hidden,
        dropout_rate=0.0
    )
    mixer_layer_instance.eval()

    # Test MixerLayer with seq_len < configured max_seq_len
    test_seq_len_mixer_short = 16
    dummy_input_mixer_short = torch.randn(batch_size, test_seq_len_mixer_short, m_dim)
    output_mixer_short = mixer_layer_instance(dummy_input_mixer_short)
    print(f"\nMixerLayer test (short seq_len={test_seq_len_mixer_short}, configured max_seq_len={m_seq_len})")
    print(f"Input shape: {dummy_input_mixer_short.shape}, Output shape: {output_mixer_short.shape}")
    assert output_mixer_short.shape == dummy_input_mixer_short.shape, "MixerLayer Test (short) failed shape check."
    print("MixerLayer Test (short) Passed.")

    # Test MixerLayer with seq_len == configured max_seq_len
    test_seq_len_mixer_equal = m_seq_len
    dummy_input_mixer_equal = torch.randn(batch_size, test_seq_len_mixer_equal, m_dim)
    output_mixer_equal = mixer_layer_instance(dummy_input_mixer_equal)
    print(f"\nMixerLayer test (equal seq_len={test_seq_len_mixer_equal}, configured max_seq_len={m_seq_len})")
    print(f"Input shape: {dummy_input_mixer_equal.shape}, Output shape: {output_mixer_equal.shape}")
    assert output_mixer_equal.shape == dummy_input_mixer_equal.shape, "MixerLayer Test (equal) failed shape check."
    print("MixerLayer Test (equal) Passed.")

    # Test MixerLayer with seq_len > configured max_seq_len (expects truncation)
    test_seq_len_mixer_long = 40
    dummy_input_mixer_long = torch.randn(batch_size, test_seq_len_mixer_long, m_dim)
    print("\nExpecting RuntimeWarning(s) for MixerLayer sequence length truncation...")
    with warnings.catch_warnings(record=True) as w_mixer:
        warnings.simplefilter("always")
        output_mixer_long = mixer_layer_instance(dummy_input_mixer_long)
        
        assert len(w_mixer) > 0, "MixerLayer Test (long) failed: No warning for truncation."
        # MixerLayer issues two warnings if current_seq_len > max_seq_len. Check for the truncation one.
        assert any("truncated" in str(warn.message).lower() for warn in w_mixer if issubclass(warn.category, RuntimeWarning)), \
            "MixerLayer Test (long) failed: No truncation RuntimeWarning message."
        print(f"Caught {len(w_mixer)} warning(s) for MixerLayer (long sequence).")

    print(f"MixerLayer test (long seq_len={test_seq_len_mixer_long}, configured max_seq_len={m_seq_len})")
    print(f"Input shape: {dummy_input_mixer_long.shape}, Output shape: {output_mixer_long.shape}")
    # Output sequence length will be truncated to m_seq_len by the padding/slicing logic for token_mix
    expected_output_shape_mixer_long = (batch_size, m_seq_len, m_dim)
    assert output_mixer_long.shape == expected_output_shape_mixer_long, "MixerLayer Test (long) failed shape check after truncation."
    print("MixerLayer Test (long) Passed (Warnings caught and output shape correct after truncation).")
    print("\n--- MixerLayer tests completed ---")
    print("\n<<<<< ALL TESTS IN mixer.py COMPLETED SUCCESSFULLY >>>>>")

