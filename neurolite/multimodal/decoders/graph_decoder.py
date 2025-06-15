"""
Décodeur spécialisé pour la génération de graphes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union, Tuple, Any
from dataclasses import dataclass
from ...core.ssm import SSMLayer

from ...Configs.config import MMGraphDecoderConfig
from .base_decoder import BaseDecoder


class GraphDecoderLayer(nn.Module):
    """
    Couche de décodeur de graphe basée sur l'attention.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        use_residual: bool = True
    ):
        """
        Initialise une couche de décodeur de graphe.
        
        Args:
            hidden_dim: Dimension cachée
            num_heads: Nombre de têtes d'attention
            dropout_rate: Taux de dropout
            use_residual: Si True, utilise des connexions résiduelles
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        
        # Auto-attention multi-têtes
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Normalisation
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Applique la couche de décodeur de graphe.
        
        Args:
            x: Caractéristiques des nœuds [batch_size, num_nodes, hidden_dim]
            attention_mask: Masque d'attention [batch_size, num_nodes, num_nodes]
            
        Returns:
            Caractéristiques mises à jour [batch_size, num_nodes, hidden_dim]
        """
        # Auto-attention
        identity = x
        x = self.norm1(x)
        
        # Convertir le masque si nécessaire
        if attention_mask is not None:
            # Convertir le masque booléen en masque flottant (-inf pour les positions masquées)
            attn_mask = attention_mask.float().masked_fill(
                attention_mask == 0, float("-inf")
            ).masked_fill(attention_mask == 1, float(0.0))
        else:
            attn_mask = None
        
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            need_weights=False
        )
        
        x = self.dropout(x)
        
        if self.use_residual:
            x = x + identity
        
        # Feed-forward
        identity = x
        x = self.norm2(x)
        x = self.ffn(x)
        
        if self.use_residual:
            x = x + identity
        
        return x


class GraphDecoder(BaseDecoder):
    """
    Décode une représentation latente pour générer un graphe (matrice d'adjacence et caractéristiques des nœuds).
    """
    def __init__(self, config: MMGraphDecoderConfig):
        """
        Initialise le décodeur de graphe.
        
        Args:
            config (MMGraphDecoderConfig): La configuration pour le décodeur de graphe.
        """
        super().__init__(config)
        self.config = config
        
        # Projection initiale du vecteur latent
        self.initial_projection = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Embedding de position pour les nœuds
        self.position_embedding = nn.Parameter(torch.zeros(1, config.max_nodes, config.hidden_dim))
        nn.init.normal_(self.position_embedding, std=0.02)
        
        # Couches de décodeur
        if config.use_ssm_layers:
            ssm_layers_list = [
                SSMLayer(
                    dim=config.hidden_dim,
                    d_state=config.ssm_d_state,
                    d_conv=config.ssm_d_conv,
                    expand_factor=config.ssm_expand_factor
                ) for _ in range(config.num_layers)
            ]
            self.decoder_layers = nn.Sequential(*ssm_layers_list)
        else:
            # Décodeur basé sur Transformer
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim,
                dropout=config.dropout_rate,
                activation="gelu",
                batch_first=True,
            )
            self.decoder_layers = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=config.num_layers
            )
        
        # Têtes de prédiction
        self.node_feature_head = nn.Linear(config.embedding_dim, config.node_feature_dim)
        
        # Prédiction de la matrice d'adjacence
        self.adjacency_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialise les poids du modèle."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        latent: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Décode une représentation latente en un graphe.
        Si des cibles sont fournies, calcule aussi la perte.
        
        Args:
            latent: Représentation latente [batch_size, input_dim].
            targets: Dictionnaire contenant les données cibles du graphe:
                     - 'node_features': [batch_size, max_nodes, node_feature_dim]
                     - 'adjacency_matrix': [batch_size, max_nodes, max_nodes]
            
        Returns:
            Dictionnaire contenant :
            - 'output': Dictionnaire avec 'node_features' et 'adjacency_matrix' générés.
            - 'loss': Perte combinée si `targets` est fourni.
        """
        generated_graph = self.generate(latent)
        
        loss = torch.tensor(0.0, device=latent.device)
        if targets is not None:
            # Perte sur les caractéristiques des noeuds (MSE)
            target_nodes = targets['node_features']
            pred_nodes = generated_graph['node_features']
            # S'assurer que les dimensions correspondent
            if pred_nodes.shape[1] != target_nodes.shape[1]:
                # Pad/truncate le plus petit
                min_nodes = min(pred_nodes.shape[1], target_nodes.shape[1])
                pred_nodes = pred_nodes[:, :min_nodes, :]
                target_nodes = target_nodes[:, :min_nodes, :]
            
            node_loss = F.mse_loss(pred_nodes, target_nodes)
            
            # Perte sur la matrice d'adjacence (BCEWithLogitsLoss)
            target_adj = targets['adjacency_matrix']
            pred_adj_logits = generated_graph['adjacency_logits']
            if pred_adj_logits.shape[1] != target_adj.shape[1]:
                min_nodes = min(pred_adj_logits.shape[1], target_adj.shape[1])
                pred_adj_logits = pred_adj_logits[:, :min_nodes, :min_nodes]
                target_adj = target_adj[:, :min_nodes, :min_nodes]

            adj_loss = F.binary_cross_entropy_with_logits(pred_adj_logits, target_adj)
            
            loss = node_loss + adj_loss
        
        return {
            'output': generated_graph,
            'loss': loss
        }

    @torch.no_grad()
    def generate(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Génère un graphe (caractéristiques de nœuds et matrice d'adjacence)
        à partir d'un vecteur latent (mode inférence).
        """
        batch_size = latent.shape[0]
        
        # Projection initiale vers un état global, répété pour chaque noeud potentiel
        x = self.initial_projection(latent).unsqueeze(1).repeat(1, self.config.max_nodes, 1)
        x = x + self.position_embedding
        
        # Raffinage des représentations de noeuds
        if self.config.use_ssm_layers:
            x = self.decoder_layers(x)
        else:
            for layer in self.decoder_layers:
                x = layer(x)
        
        # Prédiction des caractéristiques des noeuds
        node_features = self.node_feature_head(x)
        
        # Prédiction de la matrice d'adjacence
        x_i = x.unsqueeze(2).repeat(1, 1, self.config.max_nodes, 1)
        x_j = x.unsqueeze(1).repeat(1, self.config.max_nodes, 1, 1)
        edge_pairs = torch.cat([x_i, x_j], dim=-1)
        
        adj_logits = self.adjacency_predictor(edge_pairs).squeeze(-1)
        
        return {
            "node_features": node_features,
            "adjacency_logits": adj_logits,
            "adjacency_matrix": torch.sigmoid(adj_logits)
        }
