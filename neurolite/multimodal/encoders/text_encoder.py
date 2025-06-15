"""
Encodeur spécialisé pour les entrées textuelles.
"""

import torch
import torch.nn as nn
import math
from typing import Union, Optional, Dict
from torch.nn import functional as F

from neurolite.core.ssm import SSMLayer
from neurolite.Configs.config import MMTextEncoderConfig
from .base_encoder import BaseEncoder

class TextEncoder(BaseEncoder):
    """
    Encodeur pour les entrées textuelles basé sur la configuration MMTextEncoderConfig.
    """
    def __init__(self, config: MMTextEncoderConfig):
        """
        Initialise l'encodeur de texte à partir d'un objet de configuration.
        """
        super().__init__(config)
        self.config = config

        # Embedding de tokens
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Encodage positionnel
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, config.max_position_embeddings, config.hidden_dim)
        )
        self.pos_drop = nn.Dropout(p=config.dropout_rate)

        # Couches d'encodage (SSM ou Transformer)
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
                encoder_layer, num_layers=config.num_layers
            )

        # Projection de sortie vers la dimension attendue par la couche de fusion
        self.output_projection = nn.Linear(config.hidden_dim, config.output_dim)
        self.layer_norm = nn.LayerNorm(config.output_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.dropout_rate)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialise les poids du modèle."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode le texte en une représentation dense.
        Args:
            inputs: Un dictionnaire contenant:
                    - 'input_ids': Tenseur d'IDs de tokens [batch_size, seq_length]
                    - 'attention_mask': Tenseur de masque d'attention [batch_size, seq_length]
        """
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        batch_size, seq_length = input_ids.shape
        
        # Créer le masque de padding pour le TransformerEncoder
        # HuggingFace: 1 = non masqué, 0 = masqué. PyTorch: True = masqué, False = non masqué.
        padding_mask = (attention_mask == 0)
        
        # Embedding des tokens et de position
        x = self.token_embedding(input_ids)
        x = x + self.positional_encoding[:, :seq_length, :]
        x = self.pos_drop(x)
        
        # Couches d'encodage
        output = self.encoder_layers(x, src_key_padding_mask=padding_mask)

        # Projection de sortie
        output = self.output_projection(output)
        output = self.layer_norm(output)
        output = self.dropout(output)

        return output

    def resize_token_embeddings(self, new_num_tokens: int):
        """
        Redimensionne la couche d'embedding pour correspondre à un nouveau
        nombre de tokens.
        """
        old_embeddings = self.token_embedding
        new_embeddings = nn.Embedding(new_num_tokens, self.config.hidden_dim).to(old_embeddings.weight.device)
        
        # Copier les anciens poids
        num_tokens_to_copy = min(old_embeddings.num_embeddings, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
        
        self.token_embedding = new_embeddings
        self.config.vocab_size = new_num_tokens
