"""
Encodeur spécialisé pour les entrées sous forme de graphes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union

from neurolite.Configs.config import MMGraphEncoderConfig
from .base_encoder import BaseEncoder
from neurolite.core.ssm import SSMLayer


class GraphAttentionLayer(nn.Module):
    """
    Couche d'attention de graphe (GAT) pour le traitement des relations entre nœuds.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 1,
        concat: bool = True,
        dropout_rate: float = 0.1,
        alpha: float = 0.2,
        residual: bool = False
    ):
        """
        Initialise une couche d'attention de graphe.
        
        Args:
            in_features: Dimension d'entrée
            out_features: Dimension de sortie
            num_heads: Nombre de têtes d'attention
            concat: Si True, concatène les sorties des têtes, sinon les moyenne
            dropout_rate: Taux de dropout
            alpha: Pente négative du LeakyReLU
            residual: Si True, ajoute une connexion résiduelle
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout_rate = dropout_rate
        self.alpha = alpha
        self.residual = residual
        
        # Si concat est True, la dimension de sortie est divisée par le nombre de têtes
        if self.concat:
            assert out_features % num_heads == 0, "out_features must be divisible by num_heads if concat=True"
            self.head_dim = out_features // num_heads
        else:
            self.head_dim = out_features
        
        # Projections linéaires pour chaque tête d'attention
        self.W = nn.Parameter(torch.zeros(num_heads, in_features, self.head_dim))
        nn.init.xavier_uniform_(self.W)
        
        # Vecteurs d'attention pour chaque tête d'attention
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a)
        
        # Activation LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Connexion résiduelle
        if residual:
            if in_features != out_features:
                self.residual_projection = nn.Linear(in_features, out_features)
            else:
                self.residual_projection = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Calcule l'attention de graphe.
        
        Args:
            x: Caractéristiques des nœuds [batch_size, num_nodes, in_features]
            adj: Matrice d'adjacence [batch_size, num_nodes, num_nodes]
            return_attention: Si True, retourne également les coefficients d'attention
            
        Returns:
            Caractéristiques mises à jour [batch_size, num_nodes, out_features]
            et coefficients d'attention si return_attention=True
        """
        batch_size, num_nodes = x.size(0), x.size(1)
        
        # Initialiser les caractéristiques de sortie
        output = []
        attention_coefficients = []
        
        # Traiter chaque tête d'attention
        for head in range(self.num_heads):
            # Projection linéaire
            Wh = torch.bmm(x, self.W[head].expand(batch_size, -1, -1))  # [B, N, head_dim]
            
            # Calculer les scores d'attention
            # Préparer les termes pour le calcul d'attention
            Wh_repeated_rows = Wh.repeat_interleave(num_nodes, dim=1)  # [B, N*N, head_dim]
            Wh_repeated_cols = Wh.repeat(1, num_nodes, 1)  # [B, N*N, head_dim]
            
            # Concaténer pour former les paires de nœuds
            attention_input = torch.cat([Wh_repeated_rows, Wh_repeated_cols], dim=2)  # [B, N*N, 2*head_dim]
            
            # Calculer les scores d'attention
            # Reformater self.a[head] pour le produit matriciel
            # La forme actuelle est [2 * head_dim], elle doit être [2 * head_dim, 1]
            a_head = self.a[head].unsqueeze(-1)  # [2 * head_dim, 1]
            
            # Maintenant, étendre pour le traitement par lots
            a_head_expanded = a_head.expand(batch_size, 2 * self.head_dim, 1)
            
            attention_scores = torch.bmm(
                attention_input,
                a_head_expanded
            ).view(batch_size, num_nodes, num_nodes)  # [B, N, N]
            
            # Appliquer le masque d'adjacence
            attention_scores = self.leakyrelu(attention_scores)
            attention_scores = attention_scores.masked_fill_(adj == 0, -9e15)
            
            # Normaliser avec softmax
            attention_probs = F.softmax(attention_scores, dim=2)
            attention_probs = self.dropout(attention_probs)
            
            # Appliquer l'attention aux caractéristiques
            h_prime = torch.bmm(attention_probs, Wh)  # [B, N, head_dim]
            
            output.append(h_prime)
            attention_coefficients.append(attention_probs)
        
        # Combiner les têtes
        if self.concat:
            output = torch.cat(output, dim=2)
            attention_coefficients = torch.stack(attention_coefficients, dim=1)  # [B, H, N, N]
        else:
            output = torch.stack(output, dim=0).mean(dim=0)
            attention_coefficients = torch.stack(attention_coefficients, dim=0).mean(dim=0)  # [B, N, N]
        
        # Ajouter la connexion résiduelle si nécessaire
        if self.residual:
            output = output + self.residual_projection(x)
        
        if return_attention:
            return output, attention_coefficients
        else:
            return output


class GraphEncoder(BaseEncoder):
    """
    Encodeur pour les entrées de graphes, utilisant GAT ou SSM,
    configuré via MMGraphEncoderConfig.
    """
    def __init__(self, config: MMGraphEncoderConfig):
        """
        Initialise l'encodeur de graphes à partir d'un objet de configuration.
        """
        super().__init__(config)
        self.config = config

        self.node_embedding = nn.Embedding(config.num_node_types, config.hidden_dim)
        
        self.node_feature_projection = nn.Sequential(
            nn.Linear(config.node_feature_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )
        
        self.layers = nn.ModuleList()
        if config.use_ssm:
            for _ in range(config.num_layers):
                self.layers.append(SSMLayer(
                    dim=config.hidden_dim,
                    d_state=config.ssm_d_state,
                    d_conv=config.ssm_d_conv,
                    expand_factor=config.ssm_expand_factor,
                    bidirectional=config.ssm_bidirectional
                ))
        else: # GAT
            for i in range(config.num_layers):
                is_last_layer = i == config.num_layers - 1
                self.layers.append(
                    GraphAttentionLayer(
                        in_features=config.hidden_dim,
                        out_features=config.hidden_dim,
                        num_heads=1 if is_last_layer else config.num_attention_heads,
                        concat=not is_last_layer,
                        dropout_rate=config.dropout_rate,
                        residual=True
                    )
                )
        
        if config.pooling_method == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(config.hidden_dim, 1),
                nn.Softmax(dim=1)
            )

        self.output_projection = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Encode un graphe.
        
        Args:
            inputs: Dictionnaire contenant:
                - 'node_features': [batch, num_nodes, node_feature_dim]
                - 'adjacency_matrix': [batch, num_nodes, num_nodes]
                - 'node_types': [batch, num_nodes] (optionnel)
                - 'node_mask': [batch, num_nodes] (optionnel, 1 pour les vrais noeuds)
        
        Returns:
            Représentation encodée du graphe [batch_size, output_dim]
        """
        node_features = inputs['node_features']
        adj = inputs['adjacency_matrix']
        node_types = inputs.get('node_types')
        node_mask = inputs.get('node_mask')

        # Projeter les caractéristiques et les types des nœuds
        x = self.node_feature_projection(node_features)
        if node_types is not None:
            x = x + self.node_embedding(node_types)

        # Appliquer les couches GAT/SSM
        for layer in self.layers:
            x = layer(x, adj) if isinstance(layer, GraphAttentionLayer) else layer(x)

        # Appliquer le pooling pour obtenir une représentation globale du graphe
        if self.config.pooling_method == "attention":
            attention_weights = self.attention_pool(x)
            if node_mask is not None:
                attention_weights = attention_weights.masked_fill(node_mask.unsqueeze(-1) == 0, -1e9)
                attention_weights = F.softmax(attention_weights, dim=1)
            pooled_output = torch.sum(x * attention_weights, dim=1)
        else: # "mean" pooling
            if node_mask is not None:
                # Moyenne uniquement sur les vrais nœuds
                masked_x = x * node_mask.unsqueeze(-1)
                pooled_output = masked_x.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                pooled_output = x.mean(dim=1)

        # Projection finale
        output = self.output_projection(pooled_output)
        
        return output
