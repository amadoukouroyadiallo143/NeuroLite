"""
Encodeur spécialisé pour les entrées audio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union
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


class AudioEncoder(nn.Module):
    """
    Encodeur pour les entrées audio, basé sur une architecture Transformer.
    """
    
    def __init__(
        self,
        output_dim: int,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        feature_type: str = "mel_spectrogram",
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.1,
        max_audio_length_ms: int = 30000,
        use_feature_projection: bool = True,
        pooling_method: str = "mean"
    ):
        """
        Initialise l'encodeur audio.
        
        Args:
            output_dim: Dimension de sortie
            sample_rate: Taux d'échantillonnage audio
            n_fft: Taille de la FFT
            hop_length: Nombre d'échantillons entre trames consécutives
            n_mels: Nombre de bandes mel
            feature_type: Type de caractéristique ("mel_spectrogram", "mfcc", "raw_spectrogram")
            embed_dim: Dimension des embeddings
            depth: Nombre de couches transformer
            num_heads: Nombre de têtes d'attention
            mlp_ratio: Ratio pour la dimension du MLP
            dropout_rate: Taux de dropout
            max_audio_length_ms: Durée maximale de l'audio en millisecondes
            use_feature_projection: Si True, projette les caractéristiques
            pooling_method: Méthode de pooling ("mean", "max", "attention")
        """
        super().__init__()
        
        self.output_dim = output_dim
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.pooling_method = pooling_method
        self.max_frames = int((max_audio_length_ms / 1000) * sample_rate / hop_length)
        
        # Extraction de caractéristiques audio
        self.feature_extractor = AudioFeatureExtractor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            feature_type=feature_type
        )
        
        # Nombre de caractéristiques
        if feature_type == "mel_spectrogram":
            feature_dim = n_mels
        elif feature_type == "mfcc":
            feature_dim = self.feature_extractor.n_mfcc
        else:  # "raw_spectrogram"
            feature_dim = n_fft // 2 + 1
        
        # Projection des caractéristiques
        if use_feature_projection:
            self.feature_projection = nn.Sequential(
                nn.Linear(feature_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.Dropout(dropout_rate)
            )
        else:
            assert feature_dim == embed_dim, f"Feature dimension ({feature_dim}) must match embed_dim ({embed_dim}) when use_feature_projection=False"
            self.feature_projection = nn.Identity()
        
        # Embeddings positionnels
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_frames, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Dropout
        self.pos_drop = nn.Dropout(p=dropout_rate)
        
        # Couches Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=depth
        )
        
        # Attention pooling
        if pooling_method == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(embed_dim, 1),
                nn.Softmax(dim=1)
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
        features = self.feature_extractor(x)  # [batch_size, n_features, time]
        
        # Transposer pour le format attendu par le Transformer
        features = features.transpose(1, 2)  # [batch_size, time, n_features]
        
        # Limiter la longueur de séquence
        if features.size(1) > self.max_frames:
            features = features[:, :self.max_frames, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_frames]
        
        # Projeter les caractéristiques
        x = self.feature_projection(features)
        
        # Ajouter les embeddings positionnels (ajuster à la longueur actuelle)
        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len, :]
        
        # Dropout
        x = self.pos_drop(x)
        
        # Encodage Transformer
        if attention_mask is not None:
            # Convertir le masque d'attention au format requis par TransformerEncoder
            padding_mask = (attention_mask == 0)
            x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        else:
            x = self.transformer_encoder(x)
        
        # Pooling
        if self.pooling_method == "mean":
            # Pooling moyen sur les trames non masquées
            if attention_mask is not None:
                x = (x * attention_mask.unsqueeze(-1)).sum(dim=1)
                x = x / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                x = x.mean(dim=1)
        elif self.pooling_method == "max":
            # Pooling max sur les trames non masquées
            if attention_mask is not None:
                # Masquer les trames de padding avec une valeur très négative
                masked_x = x.clone()
                masked_x[~attention_mask.bool().unsqueeze(-1)] = -1e9
                x = masked_x.max(dim=1)[0]
            else:
                x = x.max(dim=1)[0]
        elif self.pooling_method == "attention":
            # Pooling par attention pondérée
            weights = self.attention_pool(x)
            if attention_mask is not None:
                # Masquer les trames de padding
                weights = weights * attention_mask.unsqueeze(-1)
                weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
            x = (weights * x).sum(dim=1)
        
        # Projection finale
        x = self.output_projection(x)
        
        return x
