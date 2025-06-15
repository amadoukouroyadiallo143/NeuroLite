"""
Encodeur spécialisé pour les entrées audio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union

from neurolite.core.ssm import SSMLayer # Added import
from neurolite.Configs.config import MMAudioEncoderConfig
from .base_encoder import BaseEncoder
import math


class AudioFeatureExtractor(nn.Module):
    """
    Module d'extraction de caractéristiques audio à partir de formes d'onde brutes.
    Convertit les données audio en spectrogrammes mel ou en autres représentations.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: int = 0,
        f_max: Optional[int] = 8000,
        window_fn: str = "hann",
        normalized: bool = True,
        center: bool = True,
        use_power_spectrum: bool = True,
        power: float = 2.0,
        feature_type: str = "mel_spectrogram"
    ):
        """
        Initialise l'extracteur de caractéristiques audio.
        
        Args:
            sample_rate: Taux d'échantillonnage audio
            n_fft: Taille de la FFT
            hop_length: Nombre d'échantillons entre trames consécutives
            n_mels: Nombre de bandes mel
            f_min: Fréquence minimale
            f_max: Fréquence maximale
            window_fn: Fonction de fenêtrage
            normalized: Si True, normalise les résultats
            center: Si True, centre les trames
            use_power_spectrum: Si True, utilise le spectre de puissance
            power: Exposant pour la conversion en spectre de puissance
            feature_type: Type de caractéristique ("mel_spectrogram", "mfcc", "raw_spectrogram")
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.feature_type = feature_type
        
        # Fonction de fenêtrage
        window_dict = {
            "hann": torch.hann_window,
            "hamming": torch.hamming_window,
            "blackman": torch.blackman_window,
            "bartlett": torch.bartlett_window
        }
        self.window_fn = window_dict.get(window_fn, torch.hann_window)
        self.normalized = normalized
        self.center = center
        self.use_power_spectrum = use_power_spectrum
        self.power = power
        
        # Filtre de banque mel
        if feature_type in ["mel_spectrogram", "mfcc"]:
            self.register_buffer(
                "mel_filters",
                self._create_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)
            )
        
        # Pour MFCC, ajouter une couche DCT
        if feature_type == "mfcc":
            self.n_mfcc = 40  # Nombre standard de coefficients MFCC
            self.register_buffer("dct_matrix", self._create_dct_matrix(n_mels, self.n_mfcc))
    
    def _create_mel_filterbank(
        self, 
        sample_rate: int, 
        n_fft: int, 
        n_mels: int, 
        f_min: int, 
        f_max: Optional[int]
    ) -> torch.Tensor:
        """
        Crée une banque de filtres mel.
        """
        # Si f_max n'est pas spécifié, le définir à sample_rate/2
        if f_max is None:
            f_max = sample_rate // 2
        
        # Convertir les fréquences de Hz à mel
        def hz_to_mel(hz):
            return 2595 * math.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)
        
        # Créer des points mel également espacés
        mel_min = hz_to_mel(f_min)
        mel_max = hz_to_mel(f_max)
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        
        # Convertir les points mel en fréquence
        hz_points = torch.tensor([mel_to_hz(m) for m in mel_points])
        
        # Convertir les fréquences en bins FFT
        bin_hz = torch.linspace(0, sample_rate // 2, n_fft // 2 + 1)
        bin_edges = torch.floor((n_fft + 1) * hz_points / sample_rate).int()
        
        # Créer la banque de filtres mel
        filters = torch.zeros(n_mels, n_fft // 2 + 1)
        for i in range(n_mels):
            start, center, end = bin_edges[i:i+3]
            
            # Remplir la première moitié du triangle
            if start < center:
                filters[i, start:center+1] = torch.linspace(0, 1, center - start + 1)
            
            # Remplir la seconde moitié du triangle
            if center < end:
                filters[i, center:end+1] = torch.linspace(1, 0, end - center + 1)
        
        return filters
    
    def _create_dct_matrix(self, n_mels: int, n_mfcc: int) -> torch.Tensor:
        """
        Crée une matrice de transformation DCT (Discrete Cosine Transform).
        """
        dct_matrix = torch.zeros(n_mfcc, n_mels)
        for i in range(n_mfcc):
            for j in range(n_mels):
                dct_matrix[i, j] = torch.cos(torch.tensor(math.pi * i * (j + 0.5) / n_mels))
        
        # Normalisation
        dct_matrix.mul_(torch.sqrt(2.0 / n_mels))
        dct_matrix[0] *= 1.0 / torch.sqrt(torch.tensor(2.0))
        
        return dct_matrix
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrait des caractéristiques audio à partir de formes d'onde.
        
        Args:
            x: Formes d'onde audio [batch_size, time]
            
        Returns:
            Caractéristiques audio [batch_size, n_features, time]
        """
        batch_size = x.size(0)
        
        # Créer la fenêtre
        window = self.window_fn(self.n_fft, device=x.device)
        
        # Calculer le spectrogramme
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=self.center,
            normalized=self.normalized,
            onesided=True,
            return_complex=True
        )
        
        # Convertir en magnitude ou spectre de puissance
        if self.use_power_spectrum:
            spec = spec.abs().pow(self.power)
        else:
            spec = spec.abs()
        
        # Appliquer les filtres mel si nécessaire
        if self.feature_type == "mel_spectrogram":
            mel_spec = torch.matmul(self.mel_filters, spec)
            # Conversion logarithmique
            log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-10))
            return log_mel_spec
        
        elif self.feature_type == "mfcc":
            mel_spec = torch.matmul(self.mel_filters, spec)
            # Conversion logarithmique
            log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-10))
            # Appliquer DCT
            mfcc = torch.matmul(self.dct_matrix, log_mel_spec)
            return mfcc
        
        else:  # "raw_spectrogram"
            return spec


class AudioEncoder(BaseEncoder):
    """
    Encodeur pour les entrées audio, basé sur une architecture Transformer ou SSM,
    configuré via MMAudioEncoderConfig.
    """
    def __init__(self, config: MMAudioEncoderConfig):
        """
        Initialise l'encodeur audio à partir d'un objet de configuration.
        """
        super().__init__(config)
        self.config = config
        
        self.feature_extractor = AudioFeatureExtractor(
            sample_rate=config.sampling_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            feature_type="mel_spectrogram"
        )
        
        self.feature_projection = nn.Linear(config.n_mels, config.hidden_dim)
        
        max_frames = (config.max_audio_length_ms // 1000) * (config.sampling_rate // config.hop_length)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, config.hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=config.dropout_rate)
        
        if config.use_ssm:
            ssm_layers = [
                SSMLayer(
                    dim=config.hidden_dim,
                    d_state=config.ssm_d_state,
                    d_conv=config.ssm_d_conv,
                    expand_factor=config.ssm_expand_factor,
                    bidirectional=config.ssm_bidirectional
                ) for _ in range(config.num_layers)
            ]
            self.encoder_layers = nn.Sequential(*ssm_layers)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=int(config.hidden_dim * config.mlp_ratio),
                dropout=config.dropout_rate,
                activation=config.activation,
                batch_first=True,
                norm_first=True
            )
            self.encoder_layers = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=config.num_layers
            )
            
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
        Encode des données audio.
        
        Args:
            x: Formes d'onde audio [batch_size, time] ou [batch_size, channels, time]
            attention_mask: Masque d'attention
            
        Returns:
            Représentation encodée [batch_size, output_dim]
        """
        # Gérer le cas où l'entrée a une dimension de canal [batch_size, channels, time]
        if x.dim() == 3:
            # Si nous avons un tenseur [batch_size, channels, time], réduire les canaux
            # en prenant la moyenne ou en utilisant uniquement le premier canal
            if x.size(1) == 1:
                # Si un seul canal, éliminer simplement la dimension
                x = x.squeeze(1)  # [batch_size, time]
            else:
                # Si plusieurs canaux, prendre la moyenne
                x = x.mean(dim=1)  # [batch_size, time]
        
        # Extraire les caractéristiques audio
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.feature_projection(x)
        
        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.pos_drop(x)
        
        if attention_mask is not None:
            # S'assurer que le masque a la bonne taille
            if attention_mask.size(1) > seq_len:
                attention_mask = attention_mask[:, :seq_len]
            elif attention_mask.size(1) < seq_len:
                # Pad le masque si nécessaire
                pad_size = seq_len - attention_mask.size(1)
                attention_mask = F.pad(attention_mask, (0, pad_size), "constant", 1)

            padding_mask = attention_mask == 0
            x = self.encoder_layers(x, src_key_padding_mask=padding_mask)
        else:
            x = self.encoder_layers(x)
            
        # Pooling
        if attention_mask is not None:
            masked_x = x * attention_mask.unsqueeze(-1)
            pooled_output = masked_x.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled_output = x.mean(dim=1)
            
        output = self.output_projection(pooled_output)
        
        return output
