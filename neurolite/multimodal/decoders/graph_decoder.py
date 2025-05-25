"""
Décodeur spécialisé pour la génération de graphes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union, Tuple, Any


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


class GraphDecoder(nn.Module):
    """
    Décodeur pour la génération de graphes à partir d'une représentation latente.
    """
    
    def __init__(
        self,
        input_dim: int,
        node_feature_dim: int = 64,
        hidden_dim: int = 256,
        max_nodes: int = 32,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        use_residual: bool = True
    ):
        """
        Initialise le décodeur de graphes.
        
        Args:
            input_dim: Dimension d'entrée
            node_feature_dim: Dimension des caractéristiques des nœuds
            hidden_dim: Dimension cachée
            max_nodes: Nombre maximum de nœuds
            num_layers: Nombre de couches de décodeur
            num_heads: Nombre de têtes d'attention
            dropout_rate: Taux de dropout
            use_residual: Si True, utilise des connexions résiduelles
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.node_feature_dim = node_feature_dim
        
        # Projection initiale du vecteur latent
        self.initial_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * max_nodes),
            nn.Dropout(dropout_rate),
            nn.GELU()
        )
        
        # Embedding de position pour les nœuds
        self.position_embedding = nn.Parameter(torch.zeros(1, max_nodes, hidden_dim))
        nn.init.normal_(self.position_embedding, std=0.02)
        
        # Couches de décodeur
        self.decoder_layers = nn.ModuleList([
            GraphDecoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                use_residual=use_residual
            )
            for _ in range(num_layers)
        ])
        
        # Prédiction des caractéristiques des nœuds
        self.node_feature_predictor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, node_feature_dim)
        )
        
        # Prédiction de la matrice d'adjacence
        self.adjacency_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Prédiction du masque des nœuds (pour déterminer quels nœuds sont réellement utilisés)
        self.node_mask_predictor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Initialisation des poids
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
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Décode une représentation latente en graphe.
        
        Args:
            latent: Représentation latente [batch_size, input_dim]
            temperature: Température pour la génération
            
        Returns:
            Dictionnaire contenant:
                - node_features: Caractéristiques des nœuds [batch_size, max_nodes, node_feature_dim]
                - adjacency_matrix: Matrice d'adjacence [batch_size, max_nodes, max_nodes]
                - node_mask: Masque des nœuds [batch_size, max_nodes]
        """
        batch_size = latent.shape[0]
        
        # Projection initiale et reshaping
        x = self.initial_projection(latent)
        x = x.view(batch_size, self.max_nodes, self.hidden_dim)
        
        # Ajouter les embeddings de position
        x = x + self.position_embedding
        
        # Passer à travers les couches de décodeur
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x)
        
        # Prédire les caractéristiques des nœuds
        node_features = self.node_feature_predictor(x)
        
        # Prédire le masque des nœuds
        node_mask = self.node_mask_predictor(x).squeeze(-1)
        
        # Prédire la matrice d'adjacence
        # Créer des paires de représentations de nœuds
        node_pairs_1 = x.unsqueeze(2).expand(batch_size, self.max_nodes, self.max_nodes, self.hidden_dim)
        node_pairs_2 = x.unsqueeze(1).expand(batch_size, self.max_nodes, self.max_nodes, self.hidden_dim)
        node_pairs = torch.cat([node_pairs_1, node_pairs_2], dim=-1)
        
        # Aplatir pour passer à travers le prédicteur
        flat_pairs = node_pairs.view(batch_size * self.max_nodes * self.max_nodes, self.hidden_dim * 2)
        flat_adjacency = self.adjacency_predictor(flat_pairs).view(batch_size, self.max_nodes, self.max_nodes)
        
        # Appliquer la température
        if temperature != 1.0:
            flat_adjacency = flat_adjacency / temperature
        
        # Convertir en probabilités
        adjacency_probs = torch.sigmoid(flat_adjacency)
        
        # Assurer que la matrice d'adjacence est symétrique
        adjacency_matrix = (adjacency_probs + adjacency_probs.transpose(1, 2)) / 2
        
        # Appliquer le masque des nœuds à la matrice d'adjacence
        node_mask_matrix = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        adjacency_matrix = adjacency_matrix * node_mask_matrix
        
        return {
            "node_features": node_features,
            "adjacency_matrix": adjacency_matrix,
            "node_mask": node_mask
        }
    
    def generate(
        self,
        latent: torch.Tensor,
        temperature: float = 1.0,
        threshold: float = 0.5,
        noise_level: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Génère un graphe à partir d'une représentation latente.
        
        Args:
            latent: Représentation latente [batch_size, input_dim]
            temperature: Contrôle la variabilité des graphes générés
            threshold: Seuil pour binariser la matrice d'adjacence
            noise_level: Niveau de bruit ajouté pendant la génération
            
        Returns:
            Dictionnaire contenant:
                - node_features: Caractéristiques des nœuds [batch_size, num_nodes, node_feature_dim]
                - adjacency_matrix: Matrice d'adjacence binaire [batch_size, num_nodes, num_nodes]
                - node_mask: Masque des nœuds [batch_size, num_nodes]
        """
        batch_size = latent.shape[0]
        
        # Ajouter du bruit au vecteur latent si nécessaire
        if noise_level > 0:
            noise = torch.randn_like(latent) * noise_level
            latent = latent + noise
        
        # Générer le graphe
        graph_data = self.forward(latent, temperature)
        
        # Extraire les composants
        node_features = graph_data["node_features"]
        adjacency_probs = graph_data["adjacency_matrix"]
        node_mask = graph_data["node_mask"]
        
        # Binariser la matrice d'adjacence
        adjacency_matrix = (adjacency_probs > threshold).float()
        
        # Binariser le masque des nœuds
        binary_node_mask = (node_mask > threshold).float()
        
        # Appliquer le masque binaire
        node_mask_matrix = binary_node_mask.unsqueeze(1) * binary_node_mask.unsqueeze(2)
        adjacency_matrix = adjacency_matrix * node_mask_matrix
        
        # Compter le nombre réel de nœuds pour chaque graphe
        num_nodes = binary_node_mask.sum(dim=1).long()
        
        # Préparer les résultats
        result = {
            "node_features": node_features,
            "adjacency_matrix": adjacency_matrix,
            "node_mask": binary_node_mask,
            "num_nodes": num_nodes
        }
        
        return result
