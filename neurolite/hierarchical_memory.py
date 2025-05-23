"""
Module de mémoire hiérarchique pour NeuroLite.
Implémente une architecture de mémoire à plusieurs niveaux (court terme, long terme, persistante)
pour une meilleure rétention contextuelle et un apprentissage continu.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
from typing import Optional, Tuple, Dict, List, Union, Any
from .memory import DifferentiableMemory
from .config import NeuroLiteConfig

class HierarchicalMemory(nn.Module):
    """
    Mémoire hiérarchique à trois niveaux pour NeuroLite.
    Combine mémoire à court terme, long terme et persistante pour une
    meilleure rétention d'information et un contexte plus riche.
    """
    
    def __init__(
        self,
        config: NeuroLiteConfig,
        hidden_size: int = 256,
        short_term_size: int = 64,
        long_term_size: int = 256,
        persistent_size: int = 512,
        key_size: Optional[int] = None,
        value_size: Optional[int] = None,
        short_term_update_rate: float = 0.9,  # Mise à jour rapide
        long_term_update_rate: float = 0.1,   # Mise à jour plus lente
        persistent_update_rate: float = 0.01,  # Très lente
        attention_heads: int = 4,
        dropout_rate: float = 0.1,
        memory_forgetting_scale: float = 0.99  # Facteur d'oubli graduel
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Déterminer les dimensions des clés/valeurs
        if key_size is None:
            key_size = hidden_size
        if value_size is None:
            value_size = hidden_size
            
        # Mémoire à court terme (quelques contextes récents)
        self.short_term_memory = DifferentiableMemory(
            hidden_size=hidden_size,
            memory_size=short_term_size,
            key_size=key_size,
            value_size=value_size,
            update_rate=short_term_update_rate
        )
        
        # Mémoire à long terme (contexte conversationnel)
        self.long_term_memory = DifferentiableMemory(
            hidden_size=hidden_size,
            memory_size=long_term_size,
            key_size=key_size,
            value_size=value_size,
            update_rate=long_term_update_rate
        )
        
        # Mémoire persistante (connaissances durables, partagées entre sessions)
        self.persistent_memory = VectorMemoryStore(
            hidden_size=hidden_size,
            memory_size=persistent_size,
            value_size=value_size,
            update_rate=persistent_update_rate
        )
        
        # Projections pour les requêtes aux différentes mémoires
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        
        # Mémoire de travail (meta-mémoire pour combiner les autres)
        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_size, 3),  # Importance relative de chaque mémoire
            nn.Softmax(dim=-1)
        )
        
        # Mécanisme d'attention multi-tête pour intégrer les mémoires
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Projection de sortie
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
        # Facteur d'oubli pour simuler une mémoire réaliste
        self.forgetting_scale = memory_forgetting_scale
        self.register_buffer('last_access_time', torch.zeros(1))

        # Seuil de nouveauté pour la consolidation (configurable si besoin)
        self.novelty_threshold_ltm = getattr(config, 'novelty_threshold_ltm', 0.5) # Example threshold
        self.novelty_threshold_pm = getattr(config, 'novelty_threshold_pm', 0.6)  # Example threshold
        
    def _calculate_novelty_score(self, input_keys: torch.Tensor, memory_keys: torch.Tensor) -> float:
        """
        Calcule un score de nouveauté moyen pour les clés d'entrée par rapport aux clés mémoire.
        Args:
            input_keys: Tensor de clés d'entrée [batch_size, num_input_keys, key_dim]
            memory_keys: Tensor de clés de mémoire [num_memory_slots, key_dim]
        Returns:
            Score de nouveauté moyen (0 à 1, où 1 est très nouveau).
        """
        if memory_keys.numel() == 0 or input_keys.numel() == 0 or self.persistent_memory.active_entries == 0: # or check for active_entries for LTM if applicable
            return 1.0 # Si la mémoire est vide, tout est nouveau

        # Normaliser les clés pour la similarité cosinus
        input_keys_norm = F.normalize(input_keys, p=2, dim=-1, eps=1e-12)
        memory_keys_norm = F.normalize(memory_keys, p=2, dim=-1, eps=1e-12)

        if memory_keys_norm.ndim == 3:
            # Handles case like [1, num_slots, key_dim] or [batch, num_slots, key_dim]
            memory_keys_norm_transposed = memory_keys_norm.transpose(-2, -1)
        elif memory_keys_norm.ndim == 2:
            # Standard case [num_slots, key_dim]
            memory_keys_norm_transposed = memory_keys_norm.t()
        else:
            # This case should ideally not be reached if memory_keys is always 2D or 3D [1,N,D]
            # For safety, raise an error or handle as appropriate for other dimensions.
            raise ValueError(
                f"memory_keys_norm has unexpected ndim: {memory_keys_norm.ndim}. "
                f"Shape: {memory_keys_norm.shape}. Expected 2D or 3D."
            )

        similarities = torch.matmul(input_keys_norm, memory_keys_norm_transposed)
        
        # Prendre la similarité maximale de chaque clé d'entrée avec n'importe quelle clé mémoire
        # max_similarity_per_input: [B, N_in]
        max_similarity_per_input, _ = torch.max(similarities, dim=-1)
        
        # Score de nouveauté = 1 - similarité maximale. Moyenne sur les clés d'entrée et le batch.
        avg_novelty = 1.0 - max_similarity_per_input.mean()
        
        return avg_novelty.item()

    def forward(
        self, 
        hidden_states: torch.Tensor,
        update_memory: bool = True,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Lit et met à jour la mémoire hiérarchique, optimisée pour les séquences longues.
        
        Args:
            hidden_states: Tensor d'entrée [batch_size, seq_len, hidden_size]
            update_memory: Si True, met à jour la mémoire avec les nouvelles entrées
            attention_mask: Masque optionnel pour l'attention [batch_size, seq_len]
            
        Returns:
            Tensor enrichi par la mémoire [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Pour les séquences très longues, utiliser une stratégie de chunking
        # pour traiter la séquence par blocs et économiser de la mémoire
        max_seq_chunk = 512  # Taille maximale d'un chunk de séquence à traiter ensemble
        
        if seq_len > max_seq_chunk:
            # Traiter la séquence par chunks pour économiser de la mémoire
            num_chunks = math.ceil(seq_len / max_seq_chunk)
            outputs = []
            
            for i in range(num_chunks):
                start_idx = i * max_seq_chunk
                end_idx = min((i + 1) * max_seq_chunk, seq_len)
                chunk_states = hidden_states[:, start_idx:end_idx, :]
                chunk_mask = attention_mask[:, start_idx:end_idx] if attention_mask is not None else None
                
                # Utiliser update_memory=True uniquement pour le premier chunk ou si explicitement demandé
                # Cette stratégie évite la surreprésentation des éléments répétés dans un batch
                chunk_update = update_memory if i == 0 else False
                
                # Traiter ce chunk avec la fonction normale
                chunk_output = self._process_chunk(chunk_states, chunk_update, chunk_mask)
                outputs.append(chunk_output)
            
            # Reconstituer la séquence complète
            return torch.cat(outputs, dim=1)
        else:
            # Pour les séquences courtes, utiliser le traitement normal
            return self._process_chunk(hidden_states, update_memory, attention_mask)
    
    def _process_chunk(
        self, 
        hidden_states: torch.Tensor,
        update_memory: bool = True,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Traite un seul chunk de la séquence d'entrée.
        """
        update_mem_base_flag = update_memory
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Simuler l'effet d'oubli en fonction du temps écoulé
        current_time = time.time()
        elapsed_time = current_time - self.last_access_time.item()
        forgetting_factor = self.forgetting_scale ** min(elapsed_time / 3600, 10)  # Limité à 10h max
        self.last_access_time.fill_(current_time)
        
        # Optimisation: pour les séquences longues, échantillonner certains points pour la mise à jour
        # au lieu de traiter chaque token (réduit la charge mémoire et le calcul)
        if seq_len > 128 and update_memory:
            # Pour les séquences longues, échantillonner des points représentatifs
            # Méthode 1: Prendre des échantillons équidistants
            sample_rate = max(1, seq_len // 64)  # Au moins 1, au plus 64 échantillons
            sampled_indices = list(range(0, seq_len, sample_rate))
            
            # Méthode 2: Inclure aussi l'information du début, du milieu et de la fin
            # qui sont souvent les parties les plus importantes
            key_positions = [0, seq_len//2, seq_len-1]
            sampled_indices = sorted(list(set(sampled_indices + key_positions)))
            
            # N'échantillonner que si cela réduit réellement la charge
            if len(sampled_indices) < seq_len * 0.5:
                sample_states = hidden_states[:, sampled_indices, :]
                sample_attn_mask = attention_mask[:, sampled_indices] if attention_mask is not None else None
                
                # Projeter les requêtes seulement pour les points échantillonnés
                queries = self.query_projection(sample_states)
                update_mem = True
            else:
                # Si l'échantillonnage ne réduit pas suffisamment, traiter normalement
                queries = self.query_projection(hidden_states)
                update_mem = update_memory
        else:
            # Pour les séquences courtes, projeter les requêtes normalement
            queries = self.query_projection(hidden_states)
        
        # --- Logique de Consolidation Intelligente ---
        update_ltm_flag = False
        update_pm_flag = False

        if update_mem_base_flag: # Seulement si une mise à jour est envisagée
            # Nouveauté pour la mémoire à long terme (LTM)
            # Utiliser les clés de la LTM si elle a une structure de clé explicite,
            # sinon memory_values si les valeurs sont utilisées comme clés implicites.
            # DifferentiableMemory a self.memory_keys et self.memory_values.
            # Assumons que self.long_term_memory.memory_keys sont les bonnes clés.
            if self.long_term_memory.memory_keys.numel() > 0 : # Check if memory has keys
                # Note: DifferentiableMemory.memory_keys might be [mem_size, key_size]
                # queries are [batch, seq, key_size]
                # We need to handle batch dimension for _calculate_novelty_score
                # For simplicity, average queries over seq_len for novelty score calculation for LTM update decision
                avg_queries_for_ltm_novelty = queries.mean(dim=1, keepdim=True)
                novelty_for_ltm = self._calculate_novelty_score(avg_queries_for_ltm_novelty, self.long_term_memory.memory_keys)
                if novelty_for_ltm > self.novelty_threshold_ltm:
                    update_ltm_flag = True
            else: # LTM est vide, donc tout est nouveau
                update_ltm_flag = True

            # Nouveauté pour la mémoire persistante (PM)
            # VectorMemoryStore a self.memory_keys
            if self.persistent_memory.memory_keys.numel() > 0 and self.persistent_memory.active_entries > 0:
                avg_queries_for_pm_novelty = queries.mean(dim=1, keepdim=True) # Ou utiliser la sortie de LTM comme source
                novelty_for_pm = self._calculate_novelty_score(avg_queries_for_pm_novelty, self.persistent_memory.memory_keys[:self.persistent_memory.active_entries])
                if novelty_for_pm > self.novelty_threshold_pm:
                    update_pm_flag = True
            else: # PM est vide, donc tout est nouveau (si on décide de la mettre à jour)
                update_pm_flag = True
        
        # Lire à partir de chaque niveau de mémoire
        # La mise à jour de la mémoire à court terme est généralement plus fréquente
        short_term_output = self.short_term_memory(queries, update_memory=update_mem_base_flag)
        
        long_term_output = self.long_term_memory(queries, update_memory=update_ltm_flag)
        
        # Pour la mémoire persistante, nous passons les 'queries' originelles.
        # Une alternative serait de passer 'long_term_output' si la consolidation est strictement hiérarchique.
        # Pour l'instant, utilisons 'queries' pour la PM également.
        persistent_output = self.persistent_memory(queries, update_memory=update_pm_flag)
        
        # Appliquer l'effet d'oubli aux mémoires à long terme et persistante
        # L'oubli devrait probablement s'appliquer indépendamment du fait que de nouvelles données soient écrites ou non.
        # Et peut-être seulement si update_memory (le flag général) était True.
        if update_memory and forgetting_factor < 1.0: # Check general update_memory flag
            with torch.no_grad():
                if hasattr(self.long_term_memory, 'memory_values'):
                    self.long_term_memory.memory_values *= forgetting_factor
                if hasattr(self.persistent_memory, 'memory_values'):
                    self.persistent_memory.memory_values *= forgetting_factor
        
        # Déterminer l'importance de chaque niveau de mémoire par token
        # Appliquer self.memory_gate directement sur les queries [batch_size, seq_len, hidden_size]
        # Résultat attendu pour memory_weights : [batch_size, seq_len, 3]
        memory_weights = self.memory_gate(queries)
        
        # Combiner les sorties de mémoire selon leur importance (par token)
        # short_term_output, long_term_output, persistent_output sont [batch, seq_len, hidden_size]
        # memory_weights[:, :, 0:1] est [batch, seq_len, 1]
        # L'opération * effectuera un broadcasting correct.
        combined_memory = (
            memory_weights[..., 0:1] * short_term_output +  # Utiliser ... pour plus de généralité
            memory_weights[..., 1:2] * long_term_output +
            memory_weights[..., 2:3] * persistent_output
        )
        
        # Intégrer la mémoire combinée aux états cachés via attention
        # Optimisation: utiliser une implémentation d'attention plus efficace pour les longues séquences
        if seq_len > 512 and not self.training:
            # Pour les très longues séquences en mode inférence, utiliser une attention linéaire approchée
            # qui réduit la complexité temporelle et spatiale de O(n²) à O(n)
            attn_output = self._efficient_attention(hidden_states, combined_memory, attention_mask)
        else:
            # Utiliser l'attention standard pour les séquences plus courtes ou en mode entraînement
            attn_output, _ = self.memory_attention(
                hidden_states, combined_memory, combined_memory,
                key_padding_mask=~attention_mask if attention_mask is not None else None
            )
        
        # Projection finale et connexion résiduelle
        output = self.output_projection(attn_output)
        return hidden_states + output
        
    def _efficient_attention(self, q, kv, mask=None):
        """
        Implémentation d'attention linéaire pour les longues séquences.
        Utilise une approximation efficace de l'attention classique.
        """
        # Implémentation basée sur une approximation de type "linformer"
        batch_size, seq_len, hidden_size = q.shape
        
        # Projeter les clés et valeurs dans un espace de dimension réduite
        projection_dim = min(128, seq_len // 4)  # Réduire la dimension de projection (hyperparametre)
        
        # Générer ou récupérer les matrices de projection linéaire
        if not hasattr(self, '_proj_E') or self._proj_E.size(0) != seq_len:
            # Créer de nouvelles matrices de projection de taille adaptée
            self._proj_E = torch.randn(seq_len, projection_dim, device=q.device) / math.sqrt(projection_dim)
            self._proj_F = torch.randn(seq_len, projection_dim, device=q.device) / math.sqrt(projection_dim)
        
        # Projeter les longueurs de séquence (approximation linéaire)
        k_projected = torch.matmul(kv.transpose(1, 2), self._proj_E).transpose(1, 2)  # [batch, proj_dim, hidden]
        v_projected = torch.matmul(kv.transpose(1, 2), self._proj_F).transpose(1, 2)  # [batch, proj_dim, hidden]
        
        # Calculer l'attention standard dans l'espace projeté (complexité réduite)
        scores = torch.matmul(q, k_projected.transpose(1, 2)) / math.sqrt(hidden_size)  # [batch, seq, proj_dim]
        
        # Appliquer le masque si présent
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        
        # Softmax et multiplication par les valeurs projetées
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v_projected)
        
        return output
        
    def reset_short_term(self):
        """Réinitialise la mémoire à court terme"""
        if hasattr(self.short_term_memory, 'reset'):
            self.short_term_memory.reset()
        else:
            # Réinitialisation manuelle
            for name, param in self.short_term_memory.named_parameters():
                if 'memory_values' in name or 'memory_keys' in name:
                    nn.init.zeros_(param)
            
    def save_persistent_memory(self, path: str):
        """Sauvegarde la mémoire persistante sur disque"""
        if hasattr(self.persistent_memory, 'save'):
            self.persistent_memory.save(path)
        else:
            # Sauvegarde manuelle des tenseurs
            state_dict = {
                'memory_keys': self.persistent_memory.memory_keys,
                'memory_values': self.persistent_memory.memory_values,
                'memory_usage': self.persistent_memory.memory_usage
                    if hasattr(self.persistent_memory, 'memory_usage') else None
            }
            torch.save(state_dict, path)
            
    def load_persistent_memory(self, path: str):
        """Charge la mémoire persistante depuis le disque"""
        if hasattr(self.persistent_memory, 'load'):
            self.persistent_memory.load(path)
        else:
            # Chargement manuel des tenseurs
            state_dict = torch.load(path)
            if hasattr(self.persistent_memory, 'memory_keys'):
                self.persistent_memory.memory_keys = state_dict['memory_keys']
            if hasattr(self.persistent_memory, 'memory_values'):
                self.persistent_memory.memory_values = state_dict['memory_values']
            if hasattr(self.persistent_memory, 'memory_usage') and state_dict['memory_usage'] is not None:
                self.persistent_memory.memory_usage = state_dict['memory_usage']


class VectorMemoryStore(nn.Module):
    """
    Mémoire vectorielle persistante.
    Stocke des vecteurs associatifs pour une récupération efficace.
    Optimisée pour une faible empreinte mémoire et des opérations rapides.
    """
    
    def __init__(
        self,
        hidden_size: int = 256,
        memory_size: int = 1024,
        key_size: Optional[int] = None,
        value_size: Optional[int] = None,
        update_rate: float = 0.05,
        similarity_threshold: float = 0.7
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        
        # Déterminer les dimensions des clés/valeurs
        self.key_size = key_size if key_size is not None else hidden_size
        self.value_size = value_size if value_size is not None else hidden_size
        
        # Taux de mise à jour
        self.update_rate = update_rate
        self.similarity_threshold = similarity_threshold
        
        # Projections pour la génération des clés/valeurs
        self.key_projection = nn.Linear(hidden_size, self.key_size)
        self.value_projection = nn.Linear(hidden_size, self.value_size)
        self.output_projection = nn.Linear(self.value_size, hidden_size)
        
        # Initialiser la mémoire
        self.register_buffer('memory_keys', torch.zeros(memory_size, self.key_size))
        self.register_buffer('memory_values', torch.zeros(memory_size, self.value_size))
        self.register_buffer('memory_usage', torch.zeros(memory_size))
        
        # Compteur d'entrées actives
        self.active_entries = 0
        
        # Métadonnées associées (facultatif)
        self.metadata = [None] * memory_size
        
        # Temperature for attention
        self.temperature = 0.1
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        update_memory: bool = True
    ) -> torch.Tensor:
        """
        Effectue une recherche et mise à jour dans la mémoire vectorielle.
        
        Args:
            hidden_states: Tensor d'entrée [batch_size, seq_len, hidden_size]
            update_memory: Si True, met à jour la mémoire avec les nouvelles entrées
            
        Returns:
            Tensor des valeurs récupérées [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Initialize a counter for debug prints if it doesn't exist
        if not hasattr(self, '_pm_forward_call_count'):
            self._pm_forward_call_count = 0
        self._pm_forward_call_count += 1
        debug_print = False  # Debug print disabled

        # Project queries to key dimension
        # hidden_states is [B, S, H]
        query_keys_proj = self.key_projection(hidden_states)  # [B, S, Dk]
        
        # if debug_print:
            #     print(f"[PMem fwd {self._pm_forward_call_count}] query_keys_proj (shape {query_keys_proj.shape}, sample):")
        #     if batch_size > 0 and seq_len > 0 and query_keys_proj.numel() > 0:
        #         print(query_keys_proj[0, :1, :5])
        #     else:
        #         print('query_keys_proj is empty or too small for sample')
        #     if query_keys_proj.numel() > 0:
        #         print(f"[PMem fwd {self._pm_forward_call_count}] query_keys_proj stats: min={query_keys_proj.min():.4f}, max={query_keys_proj.max():.4f}, mean={query_keys_proj.mean():.4f}, has_nan={torch.isnan(query_keys_proj).any()}, has_inf={torch.isinf(query_keys_proj).any()}")

        # Manual L2 Normalization for query_keys
        query_keys_norm_sq = torch.sum(query_keys_proj.square(), dim=-1, keepdim=True) # [B, S, 1]
        # Debug print removed

        rsqrt_term_query = torch.rsqrt(query_keys_norm_sq + 1e-6) # [B, S, 1]
        # Debug print removed

        query_keys_normalized = query_keys_proj * rsqrt_term_query # [B, S, Dk]
        # Debug print removed

        # Manual L2 Normalization for self.memory_keys
        # self.memory_keys is [M, Dk]
        current_memory_keys = self.memory_keys.clone() # Clone earlier
        if current_memory_keys.numel() == 0:
            # Debug print removed
            # Create a correctly shaped zero tensor for matmul, ensure it's on the right device.
            # Shape for transpose then matmul: [Dk, M]. So normalized_memory_keys should be [M, Dk]
            normalized_memory_keys = torch.zeros((0, query_keys_normalized.size(-1) if query_keys_normalized.numel() > 0 else self.key_size), device=device, dtype=query_keys_normalized.dtype if query_keys_normalized.numel() > 0 else hidden_states.dtype)
        else:
            memory_keys_norm_sq = torch.sum(current_memory_keys.square(), dim=-1, keepdim=True) # [M, 1]
            rsqrt_term_mem = torch.rsqrt(memory_keys_norm_sq + 1e-6) # [M, 1]
            normalized_memory_keys = current_memory_keys * rsqrt_term_mem # [M, Dk]
        
        # Attention scores
        # query_keys_normalized: [B, S, Dk], normalized_memory_keys.t(): [Dk, M]
        if normalized_memory_keys.numel() == 0 or query_keys_normalized.numel() == 0:
             # If either is empty, matmul is not possible or will result in empty. Create scores that lead to zero attention or are masked out.
             # Shape of scores should be [B, S, M_slots_available]. If M_slots_available is 0, then scores is [B,S,0]
            attention_scores = torch.empty((batch_size, seq_len, self.memory_keys.size(0)), device=device, dtype=query_keys_normalized.dtype if query_keys_normalized.numel() > 0 else hidden_states.dtype)
            # Debug print removed
        else:
            attention_scores = torch.matmul(query_keys_normalized, normalized_memory_keys.t()) # [B, S, M]

        # Apply temperature (ensure self.temperature is defined in __init__, e.g., self.temperature = 0.1)
        attention_scores_temp = attention_scores / self.temperature

        # Apply mask
        # self.memory_usage is [M]
        mask = (self.memory_usage > 0).float().view(1, 1, -1) # Reshape to [1, 1, M] for broadcasting
        # Ensure mask is on the same device and dtype as attention_scores_temp
        mask = mask.to(device=attention_scores_temp.device, dtype=attention_scores_temp.dtype)
        
        if attention_scores_temp.size(-1) != mask.size(-1) and self.memory_keys.numel() > 0 : # only if memory not empty
            # This can happen if memory_usage has a different size than actual memory_keys slots if memory shrinks
            # For safety, adjust mask size if necessary, or ensure memory_usage always reflects current memory_keys.size(0)
            # This is a potential bug if memory_usage is not kept in sync with actual memory slots. Assuming they are synced.
            # Debug print removed
            # Fallback: if mask is larger, slice it. If smaller, this is an issue. For now, assume it's correct or memory is empty.
            if mask.size(-1) > attention_scores_temp.size(-1):
                mask = mask[:,:,:attention_scores_temp.size(-1)]

        if attention_scores_temp.numel() > 0: # Avoid operations on empty tensors if scores are empty (e.g. [B,S,0])
            attention_scores_masked = attention_scores_temp * mask + (-1e9) * (1.0 - mask)
        else:
            attention_scores_masked = attention_scores_temp # Keep it empty

        # Softmax
        if attention_scores_masked.numel() > 0 and attention_scores_masked.size(-1) > 0:
            attention_weights = F.softmax(attention_scores_masked, dim=-1) # [B, S, M]
        else: # Handle case with 0 memory slots, softmax would error or produce NaN
            attention_weights = torch.zeros_like(attention_scores_masked) # Zeros if no memory slots to attend to
            # Debug print removed

        # Retrieve values
        # self.memory_values is [M, Dv]
        # Clone self.memory_values for use in this forward pass
        current_memory_values = self.memory_values.clone()
        if attention_weights.numel() > 0 and current_memory_values.numel() > 0 and attention_weights.size(-1) == current_memory_values.size(0):
            retrieved_values = torch.matmul(attention_weights, current_memory_values) # [B, S, Dv]
        else:
            # If memory_values is empty or mismatch, create zero output of expected shape
            # Expected Dv is self.value_projection.out_features or self.output_projection.in_features
            # For simplicity, use hidden_states's last dim if value_size is not readily available
            value_dim = current_memory_values.size(-1) if current_memory_values.numel() > 0 else hidden_states.size(-1)
            retrieved_values = torch.zeros((batch_size, seq_len, value_dim), device=device, dtype=hidden_states.dtype)
            # Debug print removed

        # Mettre à jour la mémoire si demandé
        if update_memory:
            self._update_memory(hidden_states, query_keys_proj) # Pass unnormalized projected keys
            
        # Projection de sortie
        output = self.output_projection(retrieved_values)
        
        return output
    
    def _update_memory(
        self, 
        hidden_states: torch.Tensor, 
        keys: Optional[torch.Tensor] = None
    ) -> None:
        """
        Met à jour la mémoire avec de nouvelles entrées.
        
        Args:
            hidden_states: Tensor d'entrée [batch_size, seq_len, hidden_size]
            keys: Clés pré-calculées (optionnel)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Determine the keys to be processed internally for this update cycle.
        # If 'keys' (argument) is provided, clone it to avoid modifying the original tensor.
        # Otherwise, project new keys from hidden_states.
        if keys is None: # 'keys' is the argument to _update_memory
            keys_for_internal_processing = self.key_projection(hidden_states)
        else:
            keys_for_internal_processing = keys.clone() # Clone the input argument
        
        # Normalize 'keys_for_internal_processing'. This is safe as it's either new or a clone.
        keys_norm_sq_local = torch.sum(keys_for_internal_processing.square(), dim=-1, keepdim=True)
        # Ensure this multiplication assigns to keys_for_internal_processing or a new var if it's not inplace by default
        keys_for_internal_processing = keys_for_internal_processing * torch.rsqrt(keys_norm_sq_local + 1e-6)
        
        # Calculer les valeurs (values are always freshly projected)
        values = self.value_projection(hidden_states)
        
        # Aplatir les dimensions batch et seq pour traiter chaque élément
        flat_keys = keys_for_internal_processing.view(-1, self.key_size)
        flat_values = values.view(-1, self.value_size)
        num_elements = flat_keys.size(0)
        
        # Clone memory keys for consistent reads within this update cycle
        mem_keys_for_read = self.memory_keys.clone().detach()
        
        # Pour chaque élément
        for i in range(num_elements):
            key = flat_keys[i:i+1].detach()  # Detach incoming key for storage
            value = flat_values[i:i+1].detach() # Detach incoming value for storage
            
            # Use the cloned memory for similarity calculation
            # memory_keys_norm = F.normalize(mem_keys_for_read, p=2, dim=-1) # Original approach if F.normalize was used
            # Manual L2 Normalization for mem_keys_for_read
            if mem_keys_for_read.numel() == 0:
                memory_keys_norm = torch.zeros((0, self.key_size), device=mem_keys_for_read.device, dtype=mem_keys_for_read.dtype)
            else:
                mem_keys_sq = torch.sum(mem_keys_for_read.square(), dim=-1, keepdim=True)
                memory_keys_norm = mem_keys_for_read * torch.rsqrt(mem_keys_sq + 1e-6)
            
            if memory_keys_norm.ndim == 3:
                # Handles case like [1, num_slots, key_dim] or [batch, num_slots, key_dim]
                memory_keys_norm_transposed = memory_keys_norm.transpose(-2, -1)
            elif memory_keys_norm.ndim == 2:
                # Standard case [num_slots, key_dim]
                memory_keys_norm_transposed = memory_keys_norm.t()
            else:
                # This case should ideally not be reached if memory_keys is always 2D or 3D [1,N,D]
                # For safety, raise an error or handle as appropriate for other dimensions.
                raise ValueError(
                    f"memory_keys_norm has unexpected ndim: {memory_keys_norm.ndim}. "
                    f"Shape: {memory_keys_norm.shape}. Expected 2D or 3D."
                )

            similarities = torch.matmul(key, memory_keys_norm_transposed).squeeze(0)
            
            # Trouver l'entrée la plus similaire
            # If similarities is empty (e.g. memory_keys_norm was [0,Dk]), max will error.
            if similarities.numel() == 0:
                # No existing memory to compare against, so this new key/value will be added if logic proceeds
                max_sim = torch.tensor(-float('inf'), device=key.device) # Ensure it's below any threshold
                max_idx = torch.tensor(-1, device=key.device) # Invalid index
            else:
                max_sim, max_idx = torch.max(similarities, dim=0)
            
            # Si une entrée similaire existe, la mettre à jour
            if max_sim > self.similarity_threshold and self.memory_usage[max_idx] > 0:
                # Interpolation entre ancienne et nouvelle valeur
                self.memory_values[max_idx] = (
                    (1 - self.update_rate) * self.memory_values[max_idx].detach() +
                    self.update_rate * value.detach()  # Detach new value
                )
                
                # Mettre à jour la clé également
                self.memory_keys[max_idx] = (
                    (1 - self.update_rate) * self.memory_keys[max_idx].detach() +
                    self.update_rate * key.detach()  # Detach new key
                )
                
                # Incrémenter le compteur d'usage
                self.memory_usage[max_idx] += 1
            else:
                # Trouver un emplacement inutilisé ou le moins utilisé
                if self.active_entries < self.memory_size:
                    # Utiliser un nouvel emplacement
                    idx = self.active_entries
                    self.active_entries += 1
                else:
                    # Remplacer l'entrée la moins utilisée
                    idx = torch.argmin(self.memory_usage).item()
                
                # Stocker la nouvelle entrée
                keys_part1 = self.memory_keys[:idx].detach()
                keys_part2 = self.memory_keys[idx+1:].detach()
                self.memory_keys = torch.cat((keys_part1, key.detach(), keys_part2), dim=0)
                
                values_part1 = self.memory_values[:idx].detach()
                values_part2 = self.memory_values[idx+1:].detach()
                self.memory_values = torch.cat((values_part1, value.detach(), values_part2), dim=0)
                
                self.memory_usage[idx] = 1
    
    def save(self, path: str):
        """Sauvegarde la mémoire sur disque"""
        if hasattr(self, 'save'):
            self.save(path)
        else:
            # Sauvegarde manuelle des tenseurs
            state_dict = {
                'memory_keys': self.memory_keys,
                'memory_values': self.memory_values,
                'memory_usage': self.memory_usage,
                'active_entries': self.active_entries,
                'metadata': self.metadata
            }
            torch.save(state_dict, path)
        
    def load(self, path: str):
        """Charge la mémoire depuis le disque"""
        if hasattr(self, 'load'):
            self.load(path)
        else:
            # Chargement manuel des tenseurs
            state_dict = torch.load(path)
            self.memory_keys = state_dict['memory_keys']
            self.memory_values = state_dict['memory_values']
            self.memory_usage = state_dict['memory_usage']
            self.active_entries = state_dict['active_entries']
            self.metadata = state_dict['metadata']
        
    def search(
        self, 
        query: torch.Tensor, 
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recherche les entrées les plus similaires dans la mémoire.
        
        Args:
            query: Tensor de requête [batch_size, hidden_size]
            top_k: Nombre d'entrées à récupérer
            
        Returns:
            Tuple de (indices, similarités)
        """
        # Projeter la requête en clé
        query_key = self.key_projection(query)
        query_key = F.normalize(query_key, p=2, dim=-1, eps=1e-12)
        
        # Normaliser les clés en mémoire
        memory_keys_norm = F.normalize(self.memory_keys, p=2, dim=-1)

        if memory_keys_norm.ndim == 3:
            # Handles case like [1, num_slots, key_dim] or [batch, num_slots, key_dim]
            memory_keys_norm_transposed = memory_keys_norm.transpose(-2, -1)
        elif memory_keys_norm.ndim == 2:
            # Standard case [num_slots, key_dim]
            memory_keys_norm_transposed = memory_keys_norm.t()
        else:
            # This case should ideally not be reached if memory_keys is always 2D or 3D [1,N,D]
            # For safety, raise an error or handle as appropriate for other dimensions.
            raise ValueError(
                f"memory_keys_norm has unexpected ndim: {memory_keys_norm.ndim}. "
                f"Shape: {memory_keys_norm.shape}. Expected 2D or 3D."
            )

        similarities = torch.matmul(query_key, memory_keys_norm_transposed)
        
        # Masquer les entrées inutilisées
        mask = (self.memory_usage > 0).float()
        similarities = similarities * mask + (-1e9) * (1 - mask)
        
        # Récupérer les top-k
        top_similarities, top_indices = torch.topk(similarities, min(top_k, self.active_entries))
        
        return top_indices, top_similarities
        
    def clear(self):
        """Efface le contenu de la mémoire"""
        self.memory_keys.zero_()
        self.memory_values.zero_()
        self.memory_usage.zero_()
        self.active_entries = 0
        self.metadata = [None] * self.memory_size
