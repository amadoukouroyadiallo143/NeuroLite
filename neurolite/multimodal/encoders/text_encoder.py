"""
Encodeur spécialisé pour les entrées textuelles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Union

class TextEncoder(nn.Module):
    """
    Encodeur pour les entrées textuelles avec support pour les représentations
    hash-based et token-based.
    """
    
    def __init__(
        self,
        output_dim: int,
        vocab_size: int = 50000,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        max_seq_length: int = 2048,
        use_positional_encoding: bool = True,
        pooling_method: str = "mean"
    ):
        """
        Initialise l'encodeur de texte.
        
        Args:
            output_dim: Dimension de sortie
            vocab_size: Taille du vocabulaire
            embedding_dim: Dimension des embeddings
            hidden_dim: Dimension cachée
            num_layers: Nombre de couches transformer
            num_heads: Nombre de têtes d'attention
            dropout_rate: Taux de dropout
            max_seq_length: Longueur maximale de séquence
            use_positional_encoding: Si True, utilise l'encodage positionnel
            pooling_method: Méthode de pooling ("mean", "max", "cls")
        """
        super().__init__()
        
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.pooling_method = pooling_method
        
        # Embedding de tokens
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Encodage positionnel
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.positional_encoding = nn.Parameter(
                torch.zeros(1, max_seq_length, embedding_dim)
            )
            nn.init.trunc_normal_(self.positional_encoding, std=0.02)
        
        # Couche d'encodage basée sur Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Embedding [CLS] pour le pooling de type CLS
        if pooling_method == "cls":
            self.cls_embedding = nn.Parameter(torch.zeros(1, 1, embedding_dim))
            nn.init.trunc_normal_(self.cls_embedding, std=0.02)
        
        # Projection de sortie
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Encodage MinHash pour le texte non tokenisé
        self.minhash_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(
        self,
        inputs: Union[torch.Tensor, List[str]],
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode des entrées textuelles.
        
        Args:
            inputs: Entrées textuelles (tokens ids ou texte brut)
            attention_mask: Masque d'attention
            
        Returns:
            Représentation encodée [batch_size, output_dim]
        """
        batch_size = len(inputs) if isinstance(inputs, list) else inputs.shape[0]
        
        if isinstance(inputs, list):
            # Texte brut - utiliser l'encodage MinHash
            # (Cette partie serait remplacée par un vrai encodage MinHash)
            # Pour l'instant, on simule l'encodage
            dummy_embeddings = torch.randn(batch_size, self.embedding_dim, device=self.device())
            encoded = self.minhash_encoder(dummy_embeddings)
        else:
            # IDs de tokens - utiliser l'embedding standard
            encoded = self.token_embedding(inputs)
            
            # Ajouter l'encodage positionnel
            if self.use_positional_encoding:
                seq_length = encoded.size(1)
                encoded = encoded + self.positional_encoding[:, :seq_length, :]
            
            # Ajouter le token CLS si nécessaire
            if self.pooling_method == "cls":
                cls_tokens = self.cls_embedding.expand(batch_size, 1, -1)
                encoded = torch.cat([cls_tokens, encoded], dim=1)
                
                # Mettre à jour le masque d'attention si fourni
                if attention_mask is not None:
                    cls_mask = torch.ones(batch_size, 1, device=encoded.device)
                    attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
            
            # Encodage Transformer
            if attention_mask is not None:
                # Convertir le masque d'attention au format requis par TransformerEncoder
                padding_mask = attention_mask == 0
                encoded = self.transformer_encoder(encoded, src_key_padding_mask=padding_mask)
            else:
                encoded = self.transformer_encoder(encoded)
            
            # Pooling
            if self.pooling_method == "mean":
                # Pooling moyen sur les tokens non masqués
                if attention_mask is not None:
                    encoded = (encoded * attention_mask.unsqueeze(-1)).sum(dim=1)
                    encoded = encoded / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                else:
                    encoded = encoded.mean(dim=1)
            elif self.pooling_method == "max":
                # Pooling max sur les tokens non masqués
                if attention_mask is not None:
                    # Masquer les tokens de padding avec une valeur très négative
                    masked_encoded = encoded.clone()
                    masked_encoded[~attention_mask.bool().unsqueeze(-1)] = -1e9
                    encoded = masked_encoded.max(dim=1)[0]
                else:
                    encoded = encoded.max(dim=1)[0]
            elif self.pooling_method == "cls":
                # Utiliser le token CLS
                encoded = encoded[:, 0]
        
        # Projection finale
        output = self.output_projection(encoded)
        
        return output
    
    def device(self):
        """Retourne le device du modèle."""
        return next(self.parameters()).device
