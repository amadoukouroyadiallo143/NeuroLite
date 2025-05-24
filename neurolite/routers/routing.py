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
    
    def _compute_assignment(
        self, 
        router_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        
        return dispatching_indices, expert_indices, combine_weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant dans le SparseDispatcher
        
        Args:
            x: Tensor d'entrée [batch_size, seq_len, input_size]
            
        Returns:
            Tensor transformé [batch_size, seq_len, output_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Calcul des scores de routage
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Calculer l'assignation des tokens aux experts
        dispatch_idx, expert_idx, combine_weights = self._compute_assignment(router_probs)
        
        # Si aucun token n'a été assigné (cas rare), retourner des zéros
        if len(dispatch_idx) == 0:
            return torch.zeros_like(x)
        
        # Préparer le tensor de sortie
        output = torch.zeros(
            (batch_size * seq_len, self.output_size), 
            device=x.device, 
            dtype=x.dtype
        )
        
        # Aplatir l'entrée pour faciliter l'indexation
        x_flat = x.reshape(-1, self.input_size)
        
        # Pour chaque expert, traiter les tokens qui lui sont assignés en batch
        for expert_i in range(self.num_experts):
            # Masque des tokens assignés à cet expert
            expert_mask = (expert_idx == expert_i)
            if not expert_mask.any():
                continue
                
            # Récupérer les tokens et poids pour cet expert
            expert_inputs = x_flat[dispatch_idx[expert_mask]]
            expert_weights = combine_weights[expert_mask]
            
            # Passer ces tokens à travers l'expert
            expert_outputs = self.experts[expert_i](expert_inputs)
            
            # Ajouter les sorties pondérées au tensor de sortie
            for i, (idx, weight) in enumerate(zip(dispatch_idx[expert_mask], expert_weights)):
                output[idx] += expert_outputs[i] * weight
        
        # Reshaper, normaliser et appliquer dropout
        output = output.reshape(batch_size, seq_len, self.output_size)
        output = self.layer_norm(output)
        output = self.dropout(output)
        
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
                output_size=input_size,  # Même dimension d'entrée/sortie
                num_experts=num_experts,
                top_k=top_k,
                activation=activation,
                dropout_rate=dropout_rate
            )
        else:
            self.router = MixtureOfExperts(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=input_size,  # Même dimension d'entrée/sortie
                num_experts=num_experts,
                top_k=top_k,
                activation=activation,
                dropout_rate=dropout_rate
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant dans le bloc de routage dynamique
        
        Args:
            x: Tensor d'entrée [batch_size, seq_len, input_size]
            
        Returns:
            Tensor transformé [batch_size, seq_len, input_size]
        """
        # Connexion résiduelle
        residual = x
        
        # Normalisation
        x = self.norm(x)
        
        # Routage à travers les experts
        x = self.router(x)
        
        # Ajouter la connexion résiduelle
        x = x + residual
        
        return x
