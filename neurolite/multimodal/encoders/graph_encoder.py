"""
Encodeur spécialisé pour les entrées sous forme de graphes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union


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


class GraphEncoder(nn.Module):
    """
    Encodeur pour les entrées de graphes, utilisant l'attention de graphe.
    """
    
    def __init__(
        self,
        output_dim: int,
        node_feature_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        residual: bool = True,
        concat_heads: bool = True,
        readout_method: str = "mean",
        normalize_output: bool = True
    ):
        """
        Initialise l'encodeur de graphes.
        
        Args:
            output_dim: Dimension de sortie
            node_feature_dim: Dimension des caractéristiques des nœuds
            hidden_dim: Dimension cachée
            num_layers: Nombre de couches GAT
            num_heads: Nombre de têtes d'attention
            dropout_rate: Taux de dropout
            residual: Si True, ajoute des connexions résiduelles
            concat_heads: Si True, concatène les sorties des têtes
            readout_method: Méthode de readout ("mean", "sum", "max", "attention")
            normalize_output: Si True, normalise la sortie
        """
        super().__init__()
        
        self.output_dim = output_dim
        self.readout_method = readout_method
        self.normalize_output = normalize_output
        
        # Projection initiale des caractéristiques des nœuds
        self.node_projection = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Couches GAT
        self.gat_layers = nn.ModuleList()
        
        # Première couche
        self.gat_layers.append(
            GraphAttentionLayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                num_heads=num_heads,
                concat=concat_heads,
                dropout_rate=dropout_rate,
                residual=residual
            )
        )
        
        # Couches intermédiaires
        for _ in range(1, num_layers - 1):
            self.gat_layers.append(
                GraphAttentionLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    num_heads=num_heads,
                    concat=concat_heads,
                    dropout_rate=dropout_rate,
                    residual=residual
                )
            )
        
        # Dernière couche (peut avoir une seule tête ou plusieurs)
        self.gat_layers.append(
            GraphAttentionLayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                num_heads=1,
                concat=False,
                dropout_rate=dropout_rate,
                residual=residual
            )
        )
        
        # Module de readout par attention (si utilisé)
        if readout_method == "attention":
            self.attention_readout = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Softmax(dim=1)
            )
        
        # Projection de sortie
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim) if normalize_output else nn.Identity(),
            nn.Dropout(dropout_rate)
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode des graphes.
        
        Args:
            node_features: Caractéristiques des nœuds [batch_size, num_nodes, node_feature_dim]
            adjacency_matrix: Matrice d'adjacence [batch_size, num_nodes, num_nodes]
            node_mask: Masque des nœuds [batch_size, num_nodes]
            
        Returns:
            Représentation encodée [batch_size, output_dim]
        """
        batch_size, num_nodes = node_features.shape[0], node_features.shape[1]
        
        # Projection initiale des caractéristiques des nœuds
        x = self.node_projection(node_features)
        
        # Si un masque de nœuds est fourni, l'appliquer à la matrice d'adjacence
        if node_mask is not None:
            # Créer un masque pour la matrice d'adjacence
            adj_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
            adjacency_matrix = adjacency_matrix * adj_mask
        
        # Passage à travers les couches GAT
        for gat_layer in self.gat_layers:
            x = gat_layer(x, adjacency_matrix)
        
        # Readout: agréger les caractéristiques des nœuds en une représentation de graphe
        if node_mask is not None:
            # Masquer les nœuds de padding
            x = x * node_mask.unsqueeze(-1)
        
        if self.readout_method == "mean":
            # Moyenne des caractéristiques des nœuds
            if node_mask is not None:
                graph_embedding = x.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                graph_embedding = x.mean(dim=1)
        
        elif self.readout_method == "sum":
            # Somme des caractéristiques des nœuds
            graph_embedding = x.sum(dim=1)
        
        elif self.readout_method == "max":
            # Maximum des caractéristiques des nœuds
            if node_mask is not None:
                # Masquer les nœuds de padding avec une valeur très négative
                masked_x = x.clone()
                masked_x[~node_mask.bool().unsqueeze(-1)] = -1e9
                graph_embedding = masked_x.max(dim=1)[0]
            else:
                graph_embedding = x.max(dim=1)[0]
        
        elif self.readout_method == "attention":
            # Readout par attention pondérée
            weights = self.attention_readout(x)
            if node_mask is not None:
                # Masquer les nœuds de padding
                weights = weights * node_mask.unsqueeze(-1)
                weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
            graph_embedding = (weights * x).sum(dim=1)
        
        # Projection finale
        output = self.output_projection(graph_embedding)
        
        return output
