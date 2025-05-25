"""
Implémentation de quantificateurs vectoriels pour le tokenizer multimodal NeuroLite.

Ce module fournit différentes méthodes de quantification vectorielle (VQ)
pour transformer des représentations continues en tokens discrets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any


class VectorQuantizer(nn.Module):
    """
    Quantificateur vectoriel standard avec support EMA.
    
    Transforme des vecteurs continus en indices discrets en utilisant
    un codebook de vecteurs appris. Supporte les mises à jour par EMA
    pour une meilleure stabilité.
    """
    
    def __init__(
        self,
        n_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        use_ema_updates: bool = True,
        ema_decay: float = 0.99,
        restart_unused_codes: bool = True,
        threshold_ema_dead_code: float = 1e-5
    ):
        """
        Initialise le quantificateur vectoriel.
        
        Args:
            n_embeddings: Taille du codebook (nombre de codes discrets)
            embedding_dim: Dimension des embeddings
            commitment_cost: Poids de la perte de commitment
            use_ema_updates: Si True, utilise EMA pour les mises à jour du codebook
            ema_decay: Taux de decay pour EMA
            restart_unused_codes: Si True, réinitialise les codes non utilisés
            threshold_ema_dead_code: Seuil pour détecter les codes morts
        """
        super().__init__()
        
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_ema_updates = use_ema_updates
        self.restart_unused_codes = restart_unused_codes
        self.threshold_ema_dead_code = threshold_ema_dead_code
        
        # Initialiser le codebook
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_embeddings, 1.0 / n_embeddings)
        
        # Paramètres pour EMA
        if use_ema_updates:
            self.register_buffer('ema_cluster_size', torch.zeros(n_embeddings))
            self.register_buffer('ema_w', torch.zeros_like(self.embedding.weight))
            self.register_buffer('ema_initialized', torch.zeros(1, dtype=torch.bool))
            self.ema_decay = ema_decay
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantifie l'entrée x en utilisant le codebook.
        
        Args:
            x: Entrée à quantifier [batch_size, ..., embedding_dim]
            
        Returns:
            quantized: Vecteurs quantifiés [batch_size, ..., embedding_dim]
            indices: Indices des codes du codebook [batch_size, ...]
            commitment_loss: Perte de commitment
        """
        # Aplatir x pour le calcul des distances
        input_shape = x.shape
        flat_x = x.reshape(-1, self.embedding_dim)
        
        # Calculer les distances euclidiennes entre l'entrée et les embeddings
        distances = torch.sum(flat_x**2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight**2, dim=1) - \
                    2 * torch.matmul(flat_x, self.embedding.weight.t())
        
        # Trouver l'indice le plus proche pour chaque vecteur d'entrée
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Convertir les indices en one-hot pour sélectionner les embeddings
        encodings = F.one_hot(encoding_indices, self.n_embeddings).float()
        
        # Quantifier en utilisant les indices
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.reshape(input_shape)
        
        # Calculer la perte de commitment
        commitment_loss = F.mse_loss(quantized.detach(), x)
        
        # Mettre à jour le codebook si nécessaire
        if self.training:
            if self.use_ema_updates:
                self._ema_update(flat_x, encodings)
            else:
                # Mise à jour par descente de gradient standard
                # Rien à faire ici car les gradients seront propagés normalement
                pass
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        
        # Reshaper les indices pour qu'ils correspondent à la forme d'entrée sans la dim d'embedding
        indices = encoding_indices.reshape(input_shape[:-1])
        
        return quantized, indices, self.commitment_cost * commitment_loss
    
    def _ema_update(self, flat_x: torch.Tensor, encodings: torch.Tensor):
        """
        Met à jour le codebook en utilisant Exponential Moving Average (EMA).
        
        Args:
            flat_x: Entrées aplaties [n_samples, embedding_dim]
            encodings: Encodages one-hot [n_samples, n_embeddings]
        """
        # Initialiser les tampons EMA si nécessaire
        if not self.ema_initialized.item():
            self.ema_w.data.copy_(torch.matmul(encodings.t(), flat_x))
            self.ema_cluster_size.data.copy_(encodings.sum(0))
            self.ema_initialized.fill_(True)
        
        # Calculer les mises à jour EMA
        batch_cluster_size = encodings.sum(0)
        batch_ema_cluster_size = self.ema_cluster_size * self.ema_decay + \
                                 batch_cluster_size * (1 - self.ema_decay)
        
        # Somme pondérée des vecteurs d'entrée pour chaque code
        batch_ema_w = torch.matmul(encodings.t(), flat_x)
        batch_ema_w = self.ema_w * self.ema_decay + batch_ema_w * (1 - self.ema_decay)
        
        # Mettre à jour les tampons EMA
        self.ema_cluster_size.data.copy_(batch_ema_cluster_size)
        self.ema_w.data.copy_(batch_ema_w)
        
        # Normaliser les embeddings
        n = torch.sum(self.ema_cluster_size)
        cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.n_embeddings * 1e-5) * n
        
        # Mettre à jour les poids du codebook
        self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))
        
        # Redémarrer les codes non utilisés si nécessaire
        if self.restart_unused_codes:
            self._restart_dead_codes(cluster_size)
    
    def _restart_dead_codes(self, cluster_size: torch.Tensor):
        """
        Réinitialise les codes qui ne sont pas utilisés.
        
        Args:
            cluster_size: Taille des clusters [n_embeddings]
        """
        # Identifier les codes non utilisés
        dead_indices = torch.where(cluster_size < self.threshold_ema_dead_code)[0]
        
        if len(dead_indices) > 0:
            # Prendre des codes actifs au hasard pour remplacer les codes morts
            alive_indices = torch.where(cluster_size >= self.threshold_ema_dead_code)[0]
            
            if len(alive_indices) > 0:
                # Choisir aléatoirement des codes actifs à copier
                random_indices = torch.randint(0, len(alive_indices), (len(dead_indices),))
                random_alive = alive_indices[random_indices]
                
                # Ajouter du bruit aux codes copiés
                self.embedding.weight.data[dead_indices] = self.embedding.weight.data[random_alive].clone()
                self.embedding.weight.data[dead_indices] += torch.randn_like(
                    self.embedding.weight.data[dead_indices]) * 0.1
                
                # Réinitialiser les statistiques EMA pour ces codes
                self.ema_cluster_size[dead_indices] = self.ema_cluster_size[random_alive] * 0.1
                self.ema_w[dead_indices] = self.ema_w[random_alive] * 0.1


class ResidualVQ(nn.Module):
    """
    Quantificateur vectoriel résiduel (RVQ).
    
    Applique plusieurs niveaux de quantification vectorielle en séquence,
    où chaque niveau quantifie le résidu du niveau précédent pour une 
    représentation plus précise et compressée.
    """
    
    def __init__(
        self,
        dim: int,
        num_quantizers: int = 4,
        codebook_size: int = 8192,
        shared_codebook: bool = False,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99
    ):
        """
        Initialise le quantificateur vectoriel résiduel.
        
        Args:
            dim: Dimension des vecteurs d'entrée
            num_quantizers: Nombre de quantificateurs en cascade
            codebook_size: Taille de chaque codebook
            shared_codebook: Si True, partage le même codebook entre tous les niveaux
            commitment_weight: Poids de la perte de commitment
            ema_decay: Taux de decay pour EMA
        """
        super().__init__()
        
        self.dim = dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.shared_codebook = shared_codebook
        self.commitment_weight = commitment_weight
        
        # Créer les quantificateurs
        if shared_codebook:
            # Un seul codebook partagé
            self.quantizer = VectorQuantizer(
                n_embeddings=codebook_size,
                embedding_dim=dim,
                commitment_cost=commitment_weight,
                use_ema_updates=True,
                ema_decay=ema_decay
            )
        else:
            # Codebook distinct pour chaque niveau
            self.quantizers = nn.ModuleList([
                VectorQuantizer(
                    n_embeddings=codebook_size,
                    embedding_dim=dim,
                    commitment_cost=commitment_weight,
                    use_ema_updates=True,
                    ema_decay=ema_decay
                )
                for _ in range(num_quantizers)
            ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Quantifie l'entrée en utilisant plusieurs niveaux de VQ.
        
        Args:
            x: Entrée à quantifier [batch_size, ..., dim]
            
        Returns:
            quantized: Vecteurs quantifiés [batch_size, ..., dim]
            indices: Liste des indices de chaque niveau
            commitment_loss: Perte de commitment totale
        """
        batch_size = x.size(0)
        residual = x
        total_quantized = torch.zeros_like(x)
        all_indices = []
        total_commitment_loss = 0.0
        
        # Appliquer chaque niveau de quantification
        for i in range(self.num_quantizers):
            if self.shared_codebook:
                # Utiliser le codebook partagé
                quantized, indices, commitment_loss = self.quantizer(residual)
            else:
                # Utiliser le codebook spécifique à ce niveau
                quantized, indices, commitment_loss = self.quantizers[i](residual)
            
            # Mettre à jour la sortie quantifiée totale
            total_quantized = total_quantized + quantized
            
            # Calculer le résidu pour le prochain niveau
            residual = residual - quantized
            
            # Collecter les indices et la perte
            all_indices.append(indices)
            total_commitment_loss = total_commitment_loss + commitment_loss
        
        return total_quantized, all_indices, total_commitment_loss


class HierarchicalVQ(nn.Module):
    """
    Quantificateur vectoriel hiérarchique.
    
    Construit une représentation hiérarchique à plusieurs niveaux
    avec des codebooks de granularité différente.
    """
    
    def __init__(
        self,
        level_dims: List[int],
        level_codebook_sizes: List[int],
        level_commitment_costs: List[float],
        use_ema_updates: bool = True,
        ema_decay: float = 0.99
    ):
        """
        Initialise le quantificateur hiérarchique.
        
        Args:
            level_dims: Dimensions pour chaque niveau
            level_codebook_sizes: Tailles de codebook pour chaque niveau
            level_commitment_costs: Coûts de commitment pour chaque niveau
            use_ema_updates: Si True, utilise EMA pour les mises à jour
            ema_decay: Taux de decay pour EMA
        """
        super().__init__()
        
        assert len(level_dims) == len(level_codebook_sizes) == len(level_commitment_costs), \
            "Tous les paramètres de niveau doivent avoir la même longueur"
        
        self.num_levels = len(level_dims)
        
        # Créer les quantificateurs pour chaque niveau
        self.quantizers = nn.ModuleList([
            VectorQuantizer(
                n_embeddings=level_codebook_sizes[i],
                embedding_dim=level_dims[i],
                commitment_cost=level_commitment_costs[i],
                use_ema_updates=use_ema_updates,
                ema_decay=ema_decay
            )
            for i in range(self.num_levels)
        ])
        
        # Projecteurs entre les niveaux
        self.projectors = nn.ModuleList([
            nn.Linear(level_dims[i], level_dims[i+1])
            for i in range(self.num_levels - 1)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Quantifie l'entrée en utilisant une hiérarchie de codebooks.
        
        Args:
            x: Entrée à quantifier [batch_size, ..., level_dims[0]]
            
        Returns:
            quantized_levels: Liste des sorties quantifiées à chaque niveau
            indices_levels: Liste des indices à chaque niveau
            total_commitment_loss: Perte de commitment totale
        """
        current_x = x
        quantized_levels = []
        indices_levels = []
        total_commitment_loss = 0.0
        
        # Parcourir tous les niveaux
        for i in range(self.num_levels):
            # Quantifier à ce niveau
            quantized, indices, commitment_loss = self.quantizers[i](current_x)
            
            # Collecter les résultats
            quantized_levels.append(quantized)
            indices_levels.append(indices)
            total_commitment_loss = total_commitment_loss + commitment_loss
            
            # Projeter vers le niveau suivant si ce n'est pas le dernier
            if i < self.num_levels - 1:
                current_x = self.projectors[i](quantized)
        
        return quantized_levels, indices_levels, total_commitment_loss


class DualCodebookVQ(nn.Module):
    """
    Quantificateur Vectoriel à Double Codebook.
    
    Implémente un système de double codebook où un premier codebook capture
    les caractéristiques sémantiques globales et un second codebook capture
    les détails fins. Cette architecture est au cœur du noyau latent universel
    partagé de NeuroLite, permettant une représentation riche et modulaire
    des différentes modalités.
    """
    
    def __init__(
        self,
        dim: int,
        semantic_codebook_size: int = 8192,
        detail_codebook_size: int = 32768,
        semantic_dim_ratio: float = 0.5,
        commitment_weight: float = 0.25,
        use_residual_connection: bool = True,
        num_residual_layers: int = 3,
        use_context_modulation: bool = False,
        hierarchical_levels: int = 3,
        ema_decay: float = 0.99
    ):
        """
        Initialise le quantificateur à double codebook.
        
        Args:
            dim: Dimension des vecteurs d'entrée
            semantic_codebook_size: Taille du codebook sémantique
            detail_codebook_size: Taille du codebook détaillé
            semantic_dim_ratio: Ratio de la dimension allouée au codebook sémantique
            commitment_weight: Poids de la perte de commitment
            use_residual_connection: Si True, utilise une connexion résiduelle entre les codebooks
            num_residual_layers: Nombre de couches dans la quantification résiduelle
            use_context_modulation: Si True, module les détails en fonction du contexte sémantique
            hierarchical_levels: Nombre de niveaux hiérarchiques (pour les options avancées)
            ema_decay: Taux de decay pour EMA
        """
        super().__init__()
        
        self.dim = dim
        self.semantic_dim = int(dim * semantic_dim_ratio)
        self.detail_dim = dim - self.semantic_dim
        
        self.semantic_codebook_size = semantic_codebook_size
        self.detail_codebook_size = detail_codebook_size
        self.commitment_weight = commitment_weight
        self.use_residual_connection = use_residual_connection
        self.num_residual_layers = num_residual_layers
        self.use_context_modulation = use_context_modulation
        self.hierarchical_levels = hierarchical_levels
        
        # Projecteurs pour séparer les caractéristiques sémantiques et détaillées
        self.semantic_projector = nn.Linear(dim, self.semantic_dim)
        self.detail_projector = nn.Linear(dim, self.detail_dim)
        
        # Quantificateurs vectoriels pour chaque codebook
        self.semantic_quantizer = ResidualVQ(
            dim=self.semantic_dim,
            num_quantizers=num_residual_layers if use_residual_connection else 1,
            codebook_size=semantic_codebook_size,
            commitment_weight=commitment_weight,
            ema_decay=ema_decay
        )
        
        self.detail_quantizer = ResidualVQ(
            dim=self.detail_dim,
            num_quantizers=num_residual_layers if use_residual_connection else 1,
            codebook_size=detail_codebook_size,
            commitment_weight=commitment_weight,
            ema_decay=ema_decay
        )
        
        # Projecteur pour la fusion des représentations
        self.fusion_projector = nn.Linear(self.semantic_dim + self.detail_dim, dim)
        
        # Module de modulation contextuelle (facultatif)
        if use_context_modulation:
            self.context_modulation = nn.Sequential(
                nn.Linear(self.semantic_dim, self.detail_dim),
                nn.LayerNorm(self.detail_dim),
                nn.Sigmoid()
            )
        
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Quantifie l'entrée en utilisant le système de double codebook.
        
        Args:
            x: Entrée à quantifier [batch_size, ..., dim]
            
        Returns:
            Dict contenant:
                - semantic_tokens: Tokens sémantiques quantifiés
                - semantic_indices: Indices des tokens sémantiques
                - detail_tokens: Tokens détaillés quantifiés
                - detail_indices: Indices des tokens détaillés
                - refined_tokens: Tokens combinés après modulation contextuelle
                - residual_indices: Indices résiduels (si applicable)
                - output_tokens: Tokens de sortie finaux
                - commitment_loss: Perte de commitment totale
        """
        # Mémoriser la forme d'origine
        orig_shape = x.shape
        flat_shape = (-1, self.dim)
        
        # Aplatir pour la quantification
        flat_x = x.reshape(flat_shape)
        
        # Projeter vers les espaces sémantique et détaillé
        semantic_features = self.semantic_projector(flat_x)
        detail_features = self.detail_projector(flat_x)
        
        # Quantifier les caractéristiques sémantiques
        semantic_tokens, semantic_indices_list, semantic_commitment_loss = self.semantic_quantizer(semantic_features)
        
        # Quantifier les caractéristiques détaillées
        detail_tokens, detail_indices_list, detail_commitment_loss = self.detail_quantizer(detail_features)
        
        # Modulation contextuelle (facultatif)
        if self.use_context_modulation:
            # Utiliser les tokens sémantiques pour moduler les tokens détaillés
            modulation_factors = self.context_modulation(semantic_tokens)
            modulated_detail_tokens = detail_tokens * modulation_factors
        else:
            modulated_detail_tokens = detail_tokens
        
        # Fusionner les tokens sémantiques et détaillés
        combined_tokens = torch.cat([semantic_tokens, modulated_detail_tokens], dim=-1)
        refined_tokens = self.fusion_projector(combined_tokens)
        
        # Remodeler à la forme d'origine
        semantic_tokens_reshaped = semantic_tokens.reshape(*orig_shape[:-1], self.semantic_dim)
        detail_tokens_reshaped = detail_tokens.reshape(*orig_shape[:-1], self.detail_dim)
        refined_tokens_reshaped = refined_tokens.reshape(orig_shape)
        
        # Calculer la perte de commitment totale
        commitment_loss = semantic_commitment_loss + detail_commitment_loss
        
        # Extraire les indices principaux (premier niveau)
        semantic_indices = semantic_indices_list[0]
        detail_indices = detail_indices_list[0]
        
        # Préparer les sorties
        return {
            "semantic_tokens": semantic_tokens_reshaped,
            "semantic_indices": semantic_indices,
            "detail_tokens": detail_tokens_reshaped,
            "detail_indices": detail_indices,
            "refined_tokens": refined_tokens_reshaped,
            "residual_indices": semantic_indices_list[1:] + detail_indices_list[1:] if self.use_residual_connection else [],
            "output_tokens": refined_tokens_reshaped,
            "commitment_loss": commitment_loss
        }
