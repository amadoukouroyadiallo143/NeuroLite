"""
Module de projection et de génération multimodale pour NeuroLite.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import warnings

from .encoders.text_encoder import TextEncoder
from .encoders.image_encoder import ImageEncoder
from .encoders.audio_encoder import AudioEncoder
from .encoders.video_encoder import VideoEncoder
from .encoders.graph_encoder import GraphEncoder
from .decoders.text_decoder import TextDecoder
from .decoders.image_decoder import ImageDecoder
from .decoders.audio_decoder import AudioDecoder
from .decoders.video_decoder import VideoDecoder
from .decoders.graph_decoder import GraphDecoder
from neurolite.Configs.config import ModelArchitectureConfig
from neurolite.Configs.config import NeuroLiteConfig
from .decoders.base_decoder import BaseDecoder
from .decoders import (
    TextDecoder,
    ImageDecoder,
    AudioDecoder,
    VideoDecoder,
    GraphDecoder
)


class MultiModalEncoders(nn.Module):
    """
    Conteneur et dispatcheur pour les encodeurs spécifiques à chaque modalité.
    Cette classe gère l'initialisation de tous les encodeurs nécessaires
    basée sur la configuration et encode les entrées pour les modalités présentes.
    """
    def __init__(self, config: ModelArchitectureConfig):
        """
        Initialise le dictionnaire d'encodeurs.
        Args:
            config: La configuration de l'architecture du modèle qui contient
                    les sous-configurations pour chaque encodeur de modalité.
        """
        super().__init__()
        self.config = config
        self.encoders = nn.ModuleDict()

        # Initialiser les encodeurs basés sur la configuration fournie
        if config.mm_text_encoder_config:
            self.encoders['text'] = TextEncoder(config.mm_text_encoder_config)
        if config.mm_image_encoder_config:
            self.encoders['image'] = ImageEncoder(config.mm_image_encoder_config)
        if config.mm_audio_encoder_config:
            self.encoders['audio'] = AudioEncoder(config.mm_audio_encoder_config)
        if config.mm_video_encoder_config:
            self.encoders['video'] = VideoEncoder(config.mm_video_encoder_config)
        if config.mm_graph_encoder_config:
            self.encoders['graph'] = GraphEncoder(config.mm_graph_encoder_config)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode toutes les modalités présentes dans le dictionnaire d'entrée.
        Args:
            inputs: Dictionnaire où les clés sont les noms des modalités
                    (e.g., 'text', 'image') et les valeurs sont les données brutes.
        Returns:
            Dictionnaire où les clés sont les noms des modalités et les valeurs
            sont les tenseurs de caractéristiques encodées.
        """
        encoded_features = {}
        for modality, data in inputs.items():
            if modality in self.encoders and data is not None:
                encoded_features[modality] = self.encoders[modality](data)

        if not encoded_features:
            warnings.warn("Aucune modalité valide n'a été trouvée dans les entrées ou aucun encodeur correspondant n'a été configuré.")

        return encoded_features


class MultiModalDecoders(nn.Module):
    """
    Conteneur et dispatcheur pour les décodeurs spécifiques à chaque modalité.
    Cette classe gère l'initialisation de tous les décodeurs nécessaires
    basée sur la configuration et décode une représentation latente commune
    vers une modalité de sortie cible.
    """
    def __init__(self, config: ModelArchitectureConfig):
        """
        Initialise le dictionnaire de décodeurs.
        Args:
            config: La configuration de l'architecture du modèle qui contient
                    les sous-configurations pour chaque décodeur de modalité.
        """
        super().__init__()
        self.config = config
        self.decoders = nn.ModuleDict()

        # Initialiser les décodeurs basés sur la configuration fournie
        if config.mm_text_decoder_config:
            self.decoders['text'] = TextDecoder(config.mm_text_decoder_config)
        if config.mm_image_decoder_config:
            self.decoders['image'] = ImageDecoder(config.mm_image_decoder_config)
        if config.mm_audio_decoder_config:
            self.decoders['audio'] = AudioDecoder(config.mm_audio_decoder_config)
        if config.mm_video_decoder_config:
            self.decoders['video'] = VideoDecoder(config.mm_video_decoder_config)
        if config.mm_graph_decoder_config:
            self.decoders['graph'] = GraphDecoder(config.mm_graph_decoder_config)

    def forward(
        self, 
        latent_representation: torch.Tensor, 
        target_modality: str,
        targets: Optional[Any] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Décode la représentation latente vers la modalité cible.
        """
        if target_modality not in self.decoders:
            raise ValueError(f"Aucun décodeur n'est configuré pour la modalité cible : {target_modality}")
        decoder = self.decoders[target_modality]
        
        target_data = None
        if targets is not None:
            # Gérer le cas où `targets` est un dictionnaire (pourrait être utile à l'avenir)
            # et le cas où c'est un tenseur direct (notre cas d'usage actuel).
            if isinstance(targets, dict):
                target_data = targets.get(target_modality)
            else:
                target_data = targets

        return decoder(latent_representation, targets=target_data)

    def generate(
        self,
        latent_representation: torch.Tensor,
        target_modality: str,
        **kwargs
    ) -> Any:
        """
        Génère une sortie pour la modalité cible en mode inférence.
        """
        if target_modality not in self.decoders:
            raise ValueError(f"Aucun décodeur n'est configuré pour la modalité cible : {target_modality}")
        decoder = self.decoders[target_modality]
        
        # Le TextDecoder accepte maintenant **kwargs, mais on gère le décodage ici.
        tokenizer = kwargs.pop('tokenizer', None)

        # Le décodeur génère la sortie brute (IDs de tokens pour le texte)
        generated_output = decoder.generate(latent_representation, **kwargs)
        
        # Si la cible est du texte et qu'on a un tokenizer, on décode les IDs.
        if target_modality == 'text' and tokenizer is not None:
            decoded_text = tokenizer.decode_text(generated_output)
            return {target_modality: decoded_text}

        # Pour les autres modalités, on retourne la sortie brute.
        return {target_modality: generated_output}

    def get_decoder(self, modality: str) -> Optional[BaseDecoder]:
        """
        Retourne le décodeur pour une modalité donnée.
        Args:
            modality: Le nom de la modalité.
        Returns:
            Le décodeur pour la modalité donnée, ou None si aucun décodeur n'est configuré pour cette modalité.
        """
        return self.decoders.get(modality)


class CrossModalAttention(nn.Module):
    """
    Attention croisée pour fusionner des représentations de modalités.
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attn_output, _ = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=attention_mask,
            need_weights=False
        )
        return self.layer_norm(query + attn_output)


class MultimodalProjection(nn.Module):
    """
    Processe les entrées multimodales, les encode et fusionne les représentations.
    """
    def __init__(self, config: ModelArchitectureConfig):
        super().__init__()
        self.config = config
        self.encoders = MultiModalEncoders(config)
        
        self.projection_dims = {
            name: encoder.config.output_dim for name, encoder in self.encoders.encoders.items()
        }

        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, config.multimodal_hidden_dim)
            for name, dim in self.projection_dims.items()
        })

        if getattr(config, 'use_cross_modal_attention', False):
            cross_modal_hidden_dim = getattr(config, 'multimodal_hidden_dim', config.hidden_size)
            self.cross_attention_query = nn.Parameter(
                torch.randn(1, 1, cross_modal_hidden_dim)
            )
            self.cross_modal_attention = CrossModalAttention(
                hidden_size=cross_modal_hidden_dim,
                num_heads=getattr(config, 'cross_modal_num_heads', 8),
                dropout_rate=getattr(config, 'attention_probs_dropout_prob', 0.1)
            )
        else:
            self.cross_modal_attention = None

        self.final_projection = nn.Linear(
            getattr(config, 'multimodal_hidden_dim', config.hidden_size),
            getattr(config, 'multimodal_output_dim', config.hidden_size)
        )

    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        batch_size = next((t.shape[0] for t in inputs.values() if hasattr(t, 'shape')), 1)
        
        encoded_outputs = self.encoders(inputs)
        
        projected_outputs = {
            name: self.projections[name](features)
            for name, features in encoded_outputs.items()
        }

        if not projected_outputs:
            output_dim = getattr(self.config, 'multimodal_output_dim', self.config.hidden_size)
            return torch.zeros(batch_size, output_dim, device=self.final_projection.weight.device)
            
        if self.cross_modal_attention is not None:
            kv_list = [p.unsqueeze(1) for p in projected_outputs.values()]
            key_value = torch.cat(kv_list, dim=1)
            query = self.cross_attention_query.expand(batch_size, -1, -1)
            fused_features = self.cross_modal_attention(query, key_value).squeeze(1)
        else:
            # Simple average fusion if no cross-attention
            fused_features = torch.stack(list(projected_outputs.values()), dim=0).mean(dim=0)

        return self.final_projection(fused_features)


class MultimodalGeneration(nn.Module):
    """
    Wrapper pour les décodeurs multimodaux qui gère la génération
    pour différentes modalités cibles.
    """
    def __init__(self, config: ModelArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoders = MultiModalDecoders(config)

    def forward(
        self,
        latent_representation: torch.Tensor,
        target_modality: str,
        targets: Optional[Dict[str, Any]] = None
    ) -> Dict[str, torch.Tensor]:
        return self.decoders(latent_representation, target_modality, targets)

    def generate(
        self,
        latent_representation: torch.Tensor,
        target_modality: str,
        **kwargs
    ) -> Any:
        # On délègue simplement à la méthode generate du conteneur de décodeurs
        return self.decoders.generate(latent_representation, target_modality, **kwargs)