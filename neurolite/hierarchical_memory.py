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

class HierarchicalMemory(nn.Module):
    """
    Mémoire hiérarchique à trois niveaux pour NeuroLite.
    Combine mémoire à court terme, long terme et persistante pour une
    meilleure rétention d'information et un contexte plus riche.
    """
    
    def __init__(
        self,
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
            update_mem = update_memory
        
        # Optimisation: utiliser torch.no_grad pour la récupération à partir des mémoires
        # quand on n'a pas besoin des gradients pour cette opération
        with torch.set_grad_enabled(self.training):
            # Lire à partir de chaque niveau de mémoire
            short_term_output = self.short_term_memory(
                queries, update_memory=update_mem
            )
            
            # Pour les mémoires à plus long terme, on peut économiser du calcul
            # en utilisant le même mécanisme d'échantillonnage
            long_term_output = self.long_term_memory(
                queries, update_memory=update_mem
            )
            
            persistent_output = self.persistent_memory(
                queries, update_memory=update_mem
            )
        
        # Appliquer l'effet d'oubli aux mémoires à long terme et persistante
        if update_mem and forgetting_factor < 1.0:
            with torch.no_grad():
                if hasattr(self.long_term_memory, 'memory_values'):
                    self.long_term_memory.memory_values *= forgetting_factor
                if hasattr(self.persistent_memory, 'memory_values'):
                    self.persistent_memory.memory_values *= forgetting_factor
        
        # Déterminer l'importance de chaque niveau de mémoire
        # Optimisation: calculer sur un échantillon réduit de la séquence pour les longues séquences
        if seq_len > 128:
            # Réduire la charge de calcul en prenant des moyennes sur des sous-sections
            num_sections = min(32, seq_len)
            section_size = seq_len // num_sections
            section_queries = [queries[:, i*section_size:(i+1)*section_size].mean(dim=1) for i in range(num_sections)]
            section_query = torch.stack(section_queries, dim=1).mean(dim=1)
            memory_weights = self.memory_gate(section_query).unsqueeze(1)  # [batch, 1, 3]
        else:
            memory_weights = self.memory_gate(queries.mean(dim=1)).unsqueeze(1)  # [batch, 1, 3]
        
        # Combiner les sorties de mémoire selon leur importance
        combined_memory = (
            memory_weights[:, :, 0:1] * short_term_output +
            memory_weights[:, :, 1:2] * long_term_output +
            memory_weights[:, :, 2:3] * persistent_output
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
        
        # Projeter les entrées en clés et valeurs
        query_keys = self.key_projection(hidden_states)  # [batch, seq, key_size]
        
        # Normalisation L2 pour la recherche par similarité cosinus
        query_keys = F.normalize(query_keys, p=2, dim=-1)
        
        # Calculer les similarités avec les clés stockées
        memory_keys_norm = F.normalize(self.memory_keys, p=2, dim=-1)
        
        # Calculer les similarités pour toutes les entrées du batch
        # [batch, seq, memory_size]
        similarities = torch.matmul(query_keys, memory_keys_norm.t())
        
        # Appliquer une fonction de température pour accentuer les différences
        similarities = similarities / 0.1
        
        # Appliquer un masque sur les entrées non utilisées
        mask = (self.memory_usage > 0).float().unsqueeze(0).unsqueeze(0)
        similarities = similarities * mask + (-1e9) * (1 - mask)
        
        # Obtenir une distribution sur la mémoire
        attention_weights = F.softmax(similarities, dim=-1)
        
        # Récupérer les valeurs associées
        retrieved_values = torch.matmul(attention_weights, self.memory_values)
        
        # Mettre à jour la mémoire si demandé
        if update_memory:
            self._update_memory(hidden_states, query_keys)
            
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
        
        # Si les clés n'ont pas été fournies, les calculer
        if keys is None:
            keys = self.key_projection(hidden_states)
            keys = F.normalize(keys, p=2, dim=-1)
        
        # Calculer les valeurs
        values = self.value_projection(hidden_states)
        
        # Aplatir les dimensions batch et seq pour traiter chaque élément
        flat_keys = keys.view(-1, self.key_size)
        flat_values = values.view(-1, self.value_size)
        num_elements = flat_keys.size(0)
        
        # Pour chaque élément
        for i in range(num_elements):
            key = flat_keys[i:i+1]  # Garder la dimension
            value = flat_values[i:i+1]
            
            # Vérifier la similarité avec les entrées existantes
            memory_keys_norm = F.normalize(self.memory_keys, p=2, dim=-1)
            similarities = torch.matmul(key, memory_keys_norm.t()).squeeze(0)
            
            # Trouver l'entrée la plus similaire
            max_sim, max_idx = torch.max(similarities, dim=0)
            
            # Si une entrée similaire existe, la mettre à jour
            if max_sim > self.similarity_threshold and self.memory_usage[max_idx] > 0:
                # Interpolation entre ancienne et nouvelle valeur
                self.memory_values[max_idx] = (
                    (1 - self.update_rate) * self.memory_values[max_idx] +
                    self.update_rate * value
                )
                
                # Mettre à jour la clé également
                self.memory_keys[max_idx] = (
                    (1 - self.update_rate) * self.memory_keys[max_idx] +
                    self.update_rate * key
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
                self.memory_keys[idx] = key
                self.memory_values[idx] = value
                self.memory_usage[idx] = 1
    
    def save(self, path: str):
        """Sauvegarde la mémoire sur disque"""
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
        query_key = F.normalize(query_key, p=2, dim=-1)
        
        # Normaliser les clés en mémoire
        memory_keys_norm = F.normalize(self.memory_keys, p=2, dim=-1)
        
        # Calculer les similarités
        similarities = torch.matmul(query_key, memory_keys_norm.t())
        
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
