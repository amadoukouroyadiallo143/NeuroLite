"""
Module de mémoire externe pour NeuroLite.
Implémente différents types de mémoires légères et différentiables pour
augmenter les capacités de traitement contextuel sans augmenter massivement
la taille du modèle.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class DifferentiableMemory(nn.Module):
    """
    Mémoire associative différentiable s'inspirant des réseaux de Hopfield modernes
    et des Neural Turing Machines.
    
    Cette mémoire stocke des vecteurs-clés que le modèle peut lire et mettre à jour,
    permettant de conserver des informations contextuelles au-delà de la fenêtre d'entrée.
    """
    
    def __init__(
        self,
        hidden_size: int = 256,
        memory_size: int = 64,
        key_size: Optional[int] = None,
        value_size: Optional[int] = None,
        update_rate: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.memory_size = memory_size  # Nombre de slots mémoire
        self.key_size = key_size or hidden_size
        self.value_size = value_size or hidden_size
        self.update_rate = update_rate  # Taux de mise à jour de la mémoire
        self.temperature = temperature  # Temperature pour le softmax d'adressage
        
        # Projections pour générer clés et valeurs de requête
        self.query_projection = nn.Linear(hidden_size, self.key_size)
        
        # Projections pour générer clés et valeurs mémoire à partir de l'entrée
        self.key_projection = nn.Linear(hidden_size, self.key_size)
        self.value_projection = nn.Linear(hidden_size, self.value_size)
        
        # Projection finale pour la sortie
        self.output_projection = nn.Linear(self.value_size + hidden_size, hidden_size)
        
        # Initialiser la mémoire avec des zéros (sera définie lors du premier passage)
        self.register_buffer("memory_keys", torch.zeros(1, self.memory_size, self.key_size))
        self.register_buffer("memory_values", torch.zeros(1, self.memory_size, self.value_size))
        self.register_buffer("memory_age", torch.zeros(1, self.memory_size))
        
        # Flag pour savoir si la mémoire a été initialisée
        self.initialized = False
        
    def _cosine_attention(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Calcule l'attention basée sur la similarité cosinus entre les requêtes et les clés.
        Gère les cas où les dimensions de lot ne correspondent pas.
        
        Args:
            queries: Tensor de requêtes [batch_size, seq_len, hidden_size]
            keys: Tensor de clés [batch_size, memory_size, hidden_size]
        
        Returns:
            Poids d'attention [batch_size, seq_len, memory_size]
        """
        # Normaliser les vecteurs pour la similarité cosinus
        queries_norm = F.normalize(queries, p=2, dim=-1)
        keys_norm = F.normalize(keys, p=2, dim=-1)
        
        # Vérifier et adapter les dimensions de lot si nécessaire
        batch_size_q, seq_len, hidden_size = queries_norm.shape
        batch_size_k, memory_size, _ = keys_norm.shape
        
        if batch_size_q != batch_size_k:
            # Si les dimensions de lot ne correspondent pas, adapter les clés au lot de requêtes
            if batch_size_k == 1:
                # Répliquer les clés pour correspondre au lot de requêtes
                keys_norm = keys_norm.expand(batch_size_q, -1, -1)
            elif batch_size_q == 1:
                # Utiliser la moyenne des clés sur la dimension du lot
                keys_norm = keys_norm.mean(dim=0, keepdim=True)
            else:
                # Cas plus complexe: utiliser les clés uniquement du premier élément du lot
                # ou générer une erreur informative
                raise ValueError(f"Incompatible batch sizes: queries={batch_size_q}, keys={batch_size_k}")
        
        # Calculer la similarité cosinus
        # [batch_size, seq_len, memory_size]
        similarity = torch.bmm(queries_norm, keys_norm.transpose(1, 2))
        
        # Appliquer la température pour contrôler la "sharpness" de l'attention
        similarity = similarity / self.temperature
        
        # Calculer les poids d'attention via softmax
        attention_weights = F.softmax(similarity, dim=-1)
        
        return attention_weights
    
    def _initialize_memory(self, hidden_states: torch.Tensor):
        """
        Initialise la mémoire en utilisant les premiers états cachés.
        
        Args:
            hidden_states: États cachés [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        if seq_len >= self.memory_size:
            # Utiliser les premiers états de séquence comme valeurs initiales
            sample_indices = torch.linspace(
                0, seq_len-1, self.memory_size, dtype=torch.long, device=hidden_states.device
            )
            initial_states = hidden_states[:, sample_indices]
        else:
            # Répéter les états si la séquence est plus courte que la mémoire
            repeats = math.ceil(self.memory_size / seq_len)
            expanded_states = hidden_states.repeat_interleave(repeats, dim=1)
            initial_states = expanded_states[:, :self.memory_size]
        
        # Projeter les états initiaux en clés et valeurs
        initial_keys = self.key_projection(initial_states)
        initial_values = self.value_projection(initial_states)
        
        # Stocker dans la mémoire
        self.memory_keys = initial_keys
        self.memory_values = initial_values
        self.memory_age = torch.zeros(
            batch_size, self.memory_size, device=hidden_states.device
        )
        
        self.initialized = True
        
    def _update_memory(
        self, 
        input_keys: torch.Tensor, 
        input_values: torch.Tensor,
        attention_weights: torch.Tensor
    ):
        """
        Met à jour la mémoire avec les nouvelles clés et valeurs.
        
        Args:
            input_keys: Nouvelles clés [batch_size, seq_len, key_size]
            input_values: Nouvelles valeurs [batch_size, seq_len, value_size]
            attention_weights: Poids d'attention [batch_size, seq_len, memory_size]
        """
        batch_size, seq_len, _ = input_keys.shape
        
        # Calculer la moyenne des attention_weights sur la dimension seq_len
        # Cela donne la "force d'utilisation" de chaque emplacement mémoire
        usage_weights = attention_weights.mean(dim=1)  # [batch_size, memory_size]
        
        # Incrémenter l'âge de tous les emplacements mémoire
        self.memory_age = self.memory_age + 1
        
        # Politique de remplacement: remplacer les emplacements les moins utilisés et les plus anciens
        # Combiner l'utilisation inverse et l'âge pour le score de remplacement
        replacement_scores = (1.0 - usage_weights) * torch.log1p(self.memory_age)
        
        # Pour chaque nouvel élément à écrire, trouver l'emplacement de remplacement
        for b in range(batch_size):
            for s in range(min(seq_len, self.memory_size)):
                # Trouver l'emplacement avec le score le plus élevé
                _, idx = replacement_scores[b].max(dim=0)
                
                # Mettre à jour la mémoire à cet emplacement
                self.memory_keys[b, idx] = input_keys[b, s]
                self.memory_values[b, idx] = input_values[b, s]
                self.memory_age[b, idx] = 0  # Réinitialiser l'âge
                
                # Marquer cet emplacement comme utilisé pour éviter de le réutiliser immédiatement
                replacement_scores[b, idx] = float('-inf')
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        update_memory: bool = True
    ) -> torch.Tensor:
        """
        Lecture et mise à jour optionnelle de la mémoire.
        
        Args:
            hidden_states: États cachés d'entrée [batch_size, seq_len, hidden_size]
            update_memory: Si True, met à jour la mémoire avec les nouvelles entrées
            
        Returns:
            États cachés augmentés par la mémoire [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        
        # Initialiser la mémoire si c'est le premier passage
        if not self.initialized:
            self._initialize_memory(hidden_states)
            self.initialized = True
            
        # Réinitialiser la mémoire à sa taille de batch 1 si nécessaire et la répliquer
        # Cela corrige les problèmes de dimension après plusieurs passages
        if self.memory_keys.size(0) != 1:
            # Réduire à un seul batch (moyenne sur les batch précédents)
            self.memory_keys = self.memory_keys.mean(dim=0, keepdim=True)
            self.memory_values = self.memory_values.mean(dim=0, keepdim=True)
            self.memory_age = self.memory_age.mean(dim=0, keepdim=True)
            
        # Répliquer la mémoire pour correspondre à la taille du batch actuel
        if self.memory_keys.size(0) != batch_size:
            memory_keys = self.memory_keys.repeat(batch_size, 1, 1)
            memory_values = self.memory_values.repeat(batch_size, 1, 1)
        else:
            memory_keys = self.memory_keys
            memory_values = self.memory_values
        
        # Générer clés de requête à partir des états cachés
        query_keys = self.query_projection(hidden_states)
        
        # Calcul des poids d'attention entre les requêtes et la mémoire
        attention_weights = self._cosine_attention(query_keys, memory_keys)
        
        # Lecture de la mémoire: combinaison pondérée des valeurs mémoire
        # [batch_size, seq_len, memory_size] @ [batch_size, memory_size, value_size]
        try:
            retrieved_memory = torch.bmm(attention_weights, memory_values)
        except RuntimeError as e:
            print(f"Erreur dans bmm: dimensions attention_weights={attention_weights.shape}, memory_values={memory_values.shape}")
            # Essayer de réparer les dimensions si possible
            if attention_weights.size(0) != memory_values.size(0):
                # Adapter la taille du batch pour memory_values
                memory_values = memory_values[:attention_weights.size(0)] if memory_values.size(0) > attention_weights.size(0) else memory_values.repeat(attention_weights.size(0) // memory_values.size(0) + 1, 1, 1)[:attention_weights.size(0)]
                retrieved_memory = torch.bmm(attention_weights, memory_values)
            else:
                # Si les dimensions ne correspondent toujours pas, fallback sur une solution simple
                retrieved_memory = hidden_states  # Juste un fallback pour éviter l'erreur
                print("Utilisation du fallback pour éviter l'erreur de dimension")
        
        # Si mise à jour activée, générer nouvelles clés et valeurs pour la mémoire
        if update_memory and self.update_rate > 0:
            try:
                # Sélectionner un sous-ensemble des états pour la mise à jour
                # selon le taux de mise à jour
                update_size = max(1, int(seq_len * self.update_rate))
                update_indices = torch.randperm(seq_len, device=hidden_states.device)[:update_size]
                update_states = hidden_states[:, update_indices]
                    
                # Projeter en clés et valeurs
                input_keys = self.key_projection(update_states)
                input_values = self.value_projection(update_states)
                    
                # Calculer les poids d'attention pour la mise à jour
                update_attention = self._cosine_attention(input_keys, memory_keys)
                    
                # Mettre à jour la mémoire en utilisant les variables locales
                # Puis réduire à un seul batch (moyenne) pour la persistance
                updated_keys, updated_values, updated_age = self._update_memory_local(
                    input_keys, 
                    input_values, 
                    update_attention,
                    memory_keys,
                    memory_values,
                    self.memory_age.repeat(batch_size, 1) if self.memory_age.size(0) != batch_size else self.memory_age
                )
                    
                # Moyenne sur la dimension de batch pour les futures itérations
                self.memory_keys = updated_keys.mean(dim=0, keepdim=True)
                self.memory_values = updated_values.mean(dim=0, keepdim=True)
                self.memory_age = updated_age.mean(dim=0, keepdim=True)
            except Exception as e:
                print(f"Erreur lors de la mise à jour de la mémoire: {e}")
        
        # Combiner états cachés originaux et mémoire récupérée
        combined = torch.cat([hidden_states, retrieved_memory], dim=-1)
        enhanced_states = self.output_projection(combined)
        
        return enhanced_states
        
    def _update_memory_local(
            self, 
            input_keys: torch.Tensor, 
            input_values: torch.Tensor,
            attention_weights: torch.Tensor,
            memory_keys: torch.Tensor,
            memory_values: torch.Tensor,
            memory_age: torch.Tensor
        ) -> tuple:
        """
        Version locale de _update_memory qui utilise des copies locales des mémoires
        au lieu des attributs self.memory_*.
        
        Args:
            input_keys: Nouvelles clés [batch_size, seq_len, key_size]
            input_values: Nouvelles valeurs [batch_size, seq_len, value_size]
            attention_weights: Poids d'attention [batch_size, seq_len, memory_size]
            memory_keys: Clés de mémoire actuelles [batch_size, memory_size, key_size]
            memory_values: Valeurs de mémoire actuelles [batch_size, memory_size, value_size]
            memory_age: Âge des mots en mémoire [batch_size, memory_size]
            
        Returns:
            Tuple de (memory_keys, memory_values, memory_age) mis à jour
        """
        batch_size, seq_len = input_keys.shape[0], input_keys.shape[1]
        
        # Normaliser les poids d'attention pour chaque exemple et séquence
        # Nous n'utilisons que les emplacements avec les plus fortes correspondances
        _, top_indices = torch.topk(attention_weights, k=3, dim=2)
        
        # Incrémenter l'âge de tous les emplacements mémoire
        updated_age = memory_age + 1
        
        # Pour chaque batch et séquence, mettre à jour les emplacements mémoire correspondants
        updated_keys = memory_keys.clone()
        updated_values = memory_values.clone()
        
        # Pour chaque élément du batch et chaque séquence, mettre à jour les emplacements
        for b in range(batch_size):
            for s in range(seq_len):
                # Obtenir les indices des emplacements à mettre à jour
                indices = top_indices[b, s]
                
                # Mélanger nouvelles clés/valeurs avec les anciennes (moyenne mobile)
                blend_factor = 0.7  # Facteur de mélange (plus élevé = plus de nouvelles données)
                
                # Mettre à jour les clés et valeurs pour les emplacements sélectionnés
                updated_keys[b, indices] = (1 - blend_factor) * memory_keys[b, indices] + \
                                        blend_factor * input_keys[b, s].unsqueeze(0).expand(len(indices), -1)
                updated_values[b, indices] = (1 - blend_factor) * memory_values[b, indices] + \
                                        blend_factor * input_values[b, s].unsqueeze(0).expand(len(indices), -1)
                
                # Réinitialiser l'âge des emplacements mis à jour
                updated_age[b, indices] = 0
        
        return updated_keys, updated_values, updated_age

class ModernHopfieldLayer(nn.Module):
    """
    Couche basée sur les réseaux de Hopfield modernes pour le stockage et la 
    récupération associative de motifs (Ramsauer et al., 2020).
    
    Implémente une mémoire associative avec très grande capacité de stockage
    qui peut remplacer certains aspects de l'attention.
    """
    
    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        beta: float = 1.0,
        normalize: bool = True,
        update_steps: int = 1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.num_heads = num_heads
        self.beta = beta  # Facteur d'échelle d'énergie inverse (température)
        self.normalize = normalize
        self.update_steps = update_steps  # Nombre d'étapes de mise à jour Hopfield
        
        # Projections pour clés, requêtes et valeurs
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Projection de sortie
        self.output = nn.Linear(hidden_size, hidden_size)
        
        # Normalisation et dropout
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def _hopfield_retrieval(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        """
        Effectue la récupération des patterns Hopfield.
        
        Args:
            queries: Tensor de requêtes [batch_size, num_queries, num_heads, head_dim]
            keys: Tensor de clés [batch_size, num_keys, num_heads, head_dim]
            values: Tensor de valeurs [batch_size, num_keys, num_heads, head_dim]
            
        Returns:
            Patterns récupérés [batch_size, num_queries, num_heads, head_dim]
        """
        batch_size, num_queries, num_heads, head_dim = queries.shape
        _, num_keys, _, _ = keys.shape
        
        # Normaliser pour la similarité cosinus si demandé
        if self.normalize:
            queries = F.normalize(queries, p=2, dim=-1)
            keys = F.normalize(keys, p=2, dim=-1)
        
        # Initialiser l'état avec les requêtes
        state = queries
        
        # Récupération itérative
        for _ in range(self.update_steps):
            # Calculer similarités
            # [batch, num_queries, heads, 1, dim] @ [batch, 1, heads, dim, num_keys]
            # -> [batch, num_queries, heads, 1, num_keys]
            similarities = torch.einsum(
                'bqhd,bkhd->bqhk', 
                state,
                keys
            )
            
            # Appliquer le scaling et softmax
            attention = F.softmax(self.beta * similarities, dim=-1)
            
            # Récupérer les valeurs
            # [batch, num_queries, heads, num_keys] @ [batch, num_keys, heads, dim]
            # -> [batch, num_queries, heads, dim]
            retrieved = torch.einsum(
                'bqhk,bkhd->bqhd', 
                attention,
                values
            )
            
            # Mettre à jour l'état
            state = retrieved
        
        return state
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Passage avant dans la couche Hopfield.
        
        Args:
            hidden_states: Tensor d'entrée [batch_size, seq_len, hidden_size]
            
        Returns:
            Tensor transformé [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Appliquer la normalisation en entrée
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        
        # Projeter pour obtenir Q, K, V
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        
        # Reshape pour le traitement multi-têtes
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Récupération par association Hopfield
        o = self._hopfield_retrieval(q, k, v)
        
        # Reshape et projection linéaire finale
        o = o.reshape(batch_size, seq_len, self.hidden_size)
        o = self.output(o)
        o = self.dropout(o)
        
        # Connexion résiduelle
        output = residual + o
        
        return output
