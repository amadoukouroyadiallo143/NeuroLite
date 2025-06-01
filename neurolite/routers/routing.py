"""
Module de routage dynamique pour NeuroLite.
Implémente différentes stratégies de routage conditionnel pour activer seulement 
les sous-modules pertinents selon l'entrée, économisant ainsi des calculs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple, Union, Callable


class MixtureOfExperts(nn.Module):
    """
    Implémentation légère d'un Mixture-of-Experts (MoE).
    
    Ce module contient plusieurs "experts" (sous-réseaux spécialisés) et un
    routeur qui décide quels experts activer en fonction de l'entrée.
    Pour économiser des calculs, seuls les top-k experts sont utilisés
    pour chaque token.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_experts: int = 4,
        top_k: int = 2,
        activation: Union[str, Callable] = "gelu",
        noise_factor: float = 1e-2,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)  # Ne pas dépasser le nombre d'experts
        self.noise_factor = noise_factor  # Pour le "jitter" lors de l'entraînement
        
        # Sélection de la fonction d'activation
        if isinstance(activation, str):
            self.activation = {
                "gelu": F.gelu,
                "relu": F.relu,
                "silu": F.silu,
            }[activation.lower()]
        else:
            self.activation = activation
        
        # Routeur: détermine quels experts utiliser pour chaque token
        self.router = nn.Linear(input_size, num_experts)
        
        # Experts: chaque expert est un petit MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, output_size)
            )
            for _ in range(num_experts)
        ])
        
        # Normalisation et dropout
        self.layer_norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant dans le Mixture-of-Experts
        
        Args:
            x: Tensor d'entrée [batch_size, seq_len, input_size]
            
        Returns:
            Tensor transformé [batch_size, seq_len, output_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Calcul des scores de routage
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]
        
        if self.training and self.noise_factor > 0:
            # Ajouter du bruit lors de l'entraînement pour une meilleure randomisation
            router_logits += torch.randn_like(router_logits) * self.noise_factor
        
        # Sélectionner les top-k experts pour chaque token
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Trouver les top-k experts et leurs probabilités
        # [batch_size, seq_len, top_k]
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        
        # Normaliser les probabilités des top-k experts pour qu'elles somment à 1
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Préparer le tensor de sortie
        output = torch.zeros(
            (batch_size, seq_len, self.output_size), 
            device=x.device, 
            dtype=x.dtype
        )
        
        # Pour chaque expert, traiter les tokens qui l'ont sélectionné
        for expert_idx in range(self.num_experts):
            # Créer un masque pour les tokens qui ont sélectionné cet expert
            # parmi leurs top-k
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            
            # Si aucun token n'a sélectionné cet expert, passer au suivant
            if not expert_mask.any():
                continue
            
            # Sélectionner les tokens concernés
            expert_inputs = x[expert_mask]
            
            # Passer ces tokens à travers l'expert
            expert_output = self.experts[expert_idx](expert_inputs)
            
            # Pour chaque token ayant sélectionné cet expert, récupérer le poids
            # correspondant et l'appliquer à la sortie
            for b in range(batch_size):
                for s in range(seq_len):
                    if expert_mask[b, s]:
                        # Chercher où se trouve cet expert dans le top-k
                        expert_idx_in_topk = (top_k_indices[b, s] == expert_idx).nonzero(as_tuple=True)[0]
                        # Récupérer le poids correspondant
                        weight = top_k_probs[b, s, expert_idx_in_topk]
                        # Pondérer la sortie de l'expert par ce poids
                        output[b, s] += expert_output[0] * weight
        
        # Normalisation finale et dropout
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output


class SparseDispatcher(nn.Module):
    """
    Version plus efficace du Mixture-of-Experts, optimisée pour le calcul sparse.
    Au lieu de calculer séparément chaque expert, regroupe les tokens par expert
    pour un traitement par lots plus efficace.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_experts: int = 4,
        top_k: int = 2,
        activation: str = "gelu",
        capacity_factor: float = 1.5,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.capacity_factor = capacity_factor  # Contrôle la capacité max par expert
        
        # Sélection de la fonction d'activation
        if isinstance(activation, str):
            self.activation = {
                "gelu": F.gelu,
                "relu": F.relu,
                "silu": F.silu,
            }[activation.lower()]
        else:
            self.activation = activation
        
        # Routeur
        self.router = nn.Linear(input_size, num_experts)
        
        # Experts: chaque expert est un petit MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, output_size)
            )
            for _ in range(num_experts)
        ])
        
        # Normalisation et dropout
        self.layer_norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.last_expert_counts = None # For stats
    
    def _compute_assignment(
        self, 
        router_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcule l'assignation des tokens aux experts.
        
        Args:
            router_probs: Probabilités de routage [batch_size, seq_len, num_experts]
            
        Returns:
            Tuple de (dispatching_indices, expert_indices, combine_weights):
                - dispatching_indices: indices de dispatching
                - expert_indices: indices des experts pour chaque token
                - combine_weights: poids pour combiner les sorties des experts
        """
        batch_size, seq_len, num_experts = router_probs.shape
        
        # Calculer la capacité max par expert (plus de tokens peuvent être assignés
        # à un expert populaire)
        tokens_per_batch = batch_size * seq_len
        capacity = int(tokens_per_batch * self.capacity_factor / num_experts)
        capacity = max(capacity, self.top_k)  # Capacité minimale
        
        # Pour chaque token, sélectionner les top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        
        # Normaliser les probabilités des top-k experts
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Préparer les structures pour le dispatching
        # Chaque token sera envoyé à ses top-k experts
        dispatching_indices = []  # Position du token dans le lot original
        expert_indices = []       # Expert auquel le token est envoyé
        combine_weights = []      # Poids pour pondérer la sortie de l'expert
        
        # Pour chaque expert, calculer quels tokens lui sont assignés
        for expert_idx in range(num_experts):
            # Masque des tokens qui ont sélectionné cet expert dans leur top-k
            expert_mask = (top_k_indices == expert_idx)
            
            # Pour chaque occurrence de cet expert, récupérer:
            # - la position du token
            # - le poids correspondant
            positions = torch.nonzero(expert_mask, as_tuple=True)
            
            if len(positions) == 3:  # Si au moins un token a sélectionné cet expert
                batch_idx, seq_idx, k_idx = positions
                                
                # Si nécessaire, limiter le nombre de tokens par expert à la capacité
                if len(batch_idx) > capacity:
                    # Calculer la priorité de chaque token pour cet expert
                    token_priority = top_k_probs[batch_idx, seq_idx, k_idx]
                    # Sélectionner les tokens avec les priorités les plus élevées
                    _, top_indices = torch.topk(token_priority, k=capacity)
                    batch_idx = batch_idx[top_indices]
                    seq_idx = seq_idx[top_indices]
                    k_idx = k_idx[top_indices]
                
                # Ajouter les tokens à envoyer à cet expert
                for b, s, k in zip(batch_idx, seq_idx, k_idx):
                    # Calculer l'indice linéaire du token dans le batch
                    token_idx = b * seq_len + s
                    # Ajouter aux listes de dispatching
                    dispatching_indices.append(token_idx)
                    expert_indices.append(expert_idx)
                    combine_weights.append(top_k_probs[b, s, k].item())
        
        # Convertir en tensors
        if dispatching_indices:
            dispatching_indices = torch.tensor(
                dispatching_indices, device=router_probs.device
            )
            expert_indices = torch.tensor(
                expert_indices, device=router_probs.device
            )
            combine_weights = torch.tensor(
                combine_weights, device=router_probs.device
            )
        else:
            # Si aucun token n'a été assigné (cas rare), créer des tensors vides
            dispatching_indices = torch.tensor([], device=router_probs.device, dtype=torch.long)
            expert_indices = torch.tensor([], device=router_probs.device, dtype=torch.long)
            combine_weights = torch.tensor([], device=router_probs.device, dtype=torch.float)

        # For stats: count how many tokens are assigned to each expert
        expert_counts = torch.zeros(self.num_experts, device=router_probs.device, dtype=torch.long)
        if expert_indices.numel() > 0: # Ensure expert_indices is not empty
            expert_counts.scatter_add_(0, expert_indices, torch.ones_like(expert_indices, dtype=torch.long))
        
        return dispatching_indices, expert_indices, combine_weights, expert_counts
    
    def forward(self, x: torch.Tensor, return_expert_stats: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Passage avant dans le SparseDispatcher
        
        Args:
            x: Tensor d'entrée [batch_size, seq_len, input_size]
            return_expert_stats: Si True, retourne aussi des statistiques sur l'utilisation des experts.
            
        Returns:
            Tensor transformé [batch_size, seq_len, output_size]
            ou Tuple (Tensor, Dict) si return_expert_stats est True.
        """
        batch_size, seq_len, _ = x.shape
        
        # Calcul des scores de routage
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Calculer l'assignation des tokens aux experts
        dispatch_idx, expert_idx, combine_weights, expert_counts = self._compute_assignment(router_probs)
        self.last_expert_counts = expert_counts # Store for stats

        # Si aucun token n'a été assigné (cas rare), retourner des zéros
        if len(dispatch_idx) == 0:
            output = torch.zeros((batch_size, seq_len, self.output_size), device=x.device, dtype=x.dtype)
            if return_expert_stats:
                stats = {
                    "expert_counts": self.last_expert_counts,
                    "avg_active_experts_per_token": 0.0, # top_k is effectively 0
                    "total_tokens_processed": 0
                }
                return output, stats
            return output
        
        # Préparer le tensor de sortie
        output = torch.zeros(
            (batch_size * seq_len, self.output_size), 
            device=x.device, 
            dtype=x.dtype
        )
        
        # Aplatir l'entrée pour faciliter l'indexation
        x_flat = x.reshape(-1, self.input_size)
        
        # Optimisation: Utiliser scatter_add_ pour l'agrégation des sorties d'experts
        # Initialiser output à zéro
        # output est déjà initialisé à zeros [batch_size * seq_len, self.output_size]

        # Traiter les experts et agréger leurs sorties
        for i in range(self.num_experts):
            # Indices des tokens assignés à l'expert i
            assigned_tokens_indices_for_expert_i = dispatch_idx[expert_idx == i]
            
            if assigned_tokens_indices_for_expert_i.numel() > 0:
                # Récupérer les entrées pour l'expert i
                expert_inputs = x_flat[assigned_tokens_indices_for_expert_i]

                # Calculer les sorties de l'expert i
                expert_outputs = self.experts[i](expert_inputs) # [num_assigned_tokens, output_size]

                # Récupérer les poids correspondants
                current_expert_weights = combine_weights[expert_idx == i].unsqueeze(1) # [num_assigned_tokens, 1]

                # Pondérer les sorties
                weighted_expert_outputs = expert_outputs * current_expert_weights

                # Agréger en utilisant scatter_add_
                output.scatter_add_(0, assigned_tokens_indices_for_expert_i.unsqueeze(1).expand_as(weighted_expert_outputs), weighted_expert_outputs)

        output = output.reshape(batch_size, seq_len, self.output_size)
        output = self.layer_norm(output) # Apply LayerNorm to the combined outputs
        output = self.dropout(output) # Apply Dropout

        if return_expert_stats:
            # Calculate total tokens that were actually processed by any expert
            total_tokens_processed = dispatch_idx.size(0)
            # avg_active_experts_per_token is essentially self.top_k if tokens are always assigned to top_k experts
            # A more meaningful stat might be the number of unique experts that got any tokens.
            num_unique_active_experts = (self.last_expert_counts > 0).sum().item()

            stats = {
                "expert_token_counts": self.last_expert_counts.cpu().tolist(), # counts per expert
                "num_unique_active_experts": num_unique_active_experts,
                "total_tokens_routed": total_tokens_processed,
                 # Each token is routed to self.top_k experts.
                "avg_experts_per_token_if_routed": self.top_k if total_tokens_processed > 0 else 0.0,
                "capacity_per_expert_was": int(batch_size * seq_len * self.capacity_factor / self.num_experts) if self.num_experts > 0 else 0
            }
            return output, stats
        
        return output


class DynamicRoutingBlock(nn.Module):
    """
    Bloc intégrant le routage dynamique dans l'architecture NeuroLite.
    Combine un MoE avec des connexions résiduelles et normalisation.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_experts: int = 4,
        top_k: int = 2,
        dropout_rate: float = 0.1,
        activation: str = "gelu",
        use_sparse_dispatcher: bool = True,
        layer_norm_eps: float = 1e-6
    ):
        super().__init__()
        
        # Normalisation d'entrée
        self.norm = nn.LayerNorm(input_size, eps=layer_norm_eps)
        
        # Choisir entre MoE classique ou SparseDispatcher (plus efficace)
        if use_sparse_dispatcher:
            self.router = SparseDispatcher(
                input_size=input_size,
                hidden_size=hidden_size,
            output_size=input_size,  # Output size is same as input for residual connection
                num_experts=num_experts,
                top_k=top_k,
                activation=activation,
                dropout_rate=dropout_rate
            )
        else:
        # Note: MixtureOfExperts has not been updated to return_expert_stats
        # If it needs to be used with this feature, it would require similar modifications.
            self.router = MixtureOfExperts(
                input_size=input_size,
                hidden_size=hidden_size,
            output_size=input_size,
                num_experts=num_experts,
                top_k=top_k,
                activation=activation,
                dropout_rate=dropout_rate
            )
        
    def forward(self, x: torch.Tensor, return_expert_stats: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Passage avant dans le bloc de routage dynamique
        
        Args:
            x: Tensor d'entrée [batch_size, seq_len, input_size]
            return_expert_stats: Si True, retourne aussi des statistiques sur l'utilisation des experts.

        Returns:
            Tensor transformé [batch_size, seq_len, input_size]
            ou Tuple (Tensor, Dict) si return_expert_stats est True.
        """
        residual = x
        x_norm = self.norm(x)
        
        expert_output_data = {} # To store stats if returned

        if hasattr(self.router, 'forward') and 'return_expert_stats' in self.router.forward.__code__.co_varnames:
            router_output = self.router(x_norm, return_expert_stats=return_expert_stats)
            if return_expert_stats:
                x_routed, expert_output_data = router_output
            else:
                x_routed = router_output
        else: # Fallback if router doesn't support the flag (e.g. unmodified MixtureOfExperts)
            x_routed = self.router(x_norm)
            if return_expert_stats: # Provide empty stats
                 expert_output_data = {"message": "Router does not support expert stats."}

        output = x_routed + residual
        
        if return_expert_stats:
            return output, expert_output_data
        return output
