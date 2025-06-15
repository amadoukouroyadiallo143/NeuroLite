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
from neurolite.Configs.config import NeuroLiteConfig
import faiss

class HierarchicalMemory(nn.Module):
    """
    Mémoire hiérarchique à trois niveaux pour NeuroLite.
    Combine mémoire à court terme, long terme et persistante pour une
    meilleure rétention d'information et un contexte plus riche.
    """
    
    def __init__(self, config: NeuroLiteConfig):
        """
        Initialise la mémoire hiérarchique en se basant sur l'objet de configuration global.
        
        Args:
            config: Configuration complète du modèle (NeuroLiteConfig).
        """
        super().__init__()
        
        # Récupération propre des configurations
        self.config = config
        model_cfg = config.model_config
        mem_cfg = config.memory_config
        ltm_cfg = config.long_term_memory_config
        
        # --- Définition des paramètres ---
        self.hidden_size = model_cfg.hidden_size
        self.key_size = mem_cfg.memory_dim
        self.value_size = mem_cfg.memory_dim
        self.attention_heads = mem_cfg.num_memory_heads
        self.dropout_rate = model_cfg.dropout_rate
        
        # Tailles des mémoires
        short_term_size = model_cfg.short_term_memory_size
        long_term_size = ltm_cfg.memory_size
        persistent_size = model_cfg.persistent_memory_size
        
        # Taux de mise à jour
        self.short_term_update_rate = getattr(mem_cfg, 'short_term_update_rate', 0.9)
        self.long_term_update_rate = ltm_cfg.update_rate
        self.persistent_update_rate = getattr(model_cfg, 'persistent_update_rate', 0.01)
        
        # Seuils de nouveauté
        self.novelty_threshold_ltm = getattr(mem_cfg, 'novelty_threshold_ltm', 0.5)
        self.novelty_threshold_pm = ltm_cfg.similarity_threshold
        
        # Facteur d'oubli
        self.memory_forgetting_scale = getattr(mem_cfg, 'memory_forgetting_scale', 0.999)

        # --- Initialisation des modules de mémoire ---
        
        # Mémoire à court terme
        self.short_term_memory = DifferentiableMemory(
            hidden_size=self.hidden_size,
            memory_size=short_term_size,
            key_size=self.key_size,
            value_size=self.value_size,
            num_heads=self.attention_heads,
            dropout_rate=self.dropout_rate
        )
        
        # Mémoire à long terme
        self.long_term_memory = DifferentiableMemory(
            hidden_size=self.hidden_size,
            memory_size=long_term_size,
            key_size=self.key_size,
            value_size=self.value_size,
            update_rate=self.long_term_update_rate,
            num_heads=self.attention_heads,
            dropout_rate=self.dropout_rate
        )
        
        # Mémoire persistante
        self.persistent_memory = VectorMemoryStore(
            hidden_size=self.hidden_size,
            memory_size=persistent_size,
            key_size=self.key_size,
            value_size=self.value_size,
            update_rate=self.persistent_update_rate,
            similarity_threshold=self.novelty_threshold_pm
        )
        
        # --- Modules de contrôle et d'intégration ---
        
        self.query_projection = nn.Linear(self.hidden_size, self.key_size)
        
        self.memory_gate = nn.Sequential(
            nn.Linear(self.hidden_size, 3),
            nn.Softmax(dim=-1)
        )
        
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.attention_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.register_buffer('last_access_time', torch.zeros(1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialise les poids des couches linéaires."""
        for module in [self.query_projection, self.output_projection]:
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialisation spécifique pour la couche de gate
        if hasattr(self.memory_gate[0], 'weight'):
            nn.init.orthogonal_(self.memory_gate[0].weight)
            if hasattr(self.memory_gate[0], 'bias') and self.memory_gate[0].bias is not None:
                nn.init.constant_(self.memory_gate[0].bias, 0.1)

    def _search_differentiable_mem(
        self, 
        memory_component: 'DifferentiableMemory', 
        query_emb_single: torch.Tensor, 
        k_to_retrieve: int,
        similarity_exponent: float = 1.0
    ) -> List[Tuple[float, torch.Tensor]]:
        """Helper pour rechercher dans DifferentiableMemory (STM, LTM)."""
        retrieved_items: List[Tuple[float, torch.Tensor]] = []
        if not memory_component.initialized or memory_component.memory_keys.nelement() == 0:
            return retrieved_items

        try:
            projected_query = memory_component.query_projection(query_emb_single)  # [1, key_size]
            mem_keys = memory_component.memory_keys[0]  # [memory_size, key_size]
            
            similarities = None
            if mem_keys.numel() > 0 and projected_query.numel() > 0:
                similarities = F.cosine_similarity(projected_query, mem_keys, dim=-1) # [memory_size]
            
            if similarities is not None and similarities.numel() > 0:
                if similarity_exponent != 1.0:
                    # Appliquer l'exposant pour moduler les scores
                    # Clamper pour éviter les problèmes avec les nombres négatifs si cos sim peut être < 0
                    similarities = torch.clamp(similarities, 0.0, 1.0) ** similarity_exponent 

                actual_k = min(k_to_retrieve, similarities.size(0))
                if actual_k > 0:
                    scores, indices = torch.topk(similarities, actual_k)
                    for score, idx in zip(scores, indices):
                        retrieved_items.append((score.item(), memory_component.memory_values[0, idx]))
        except Exception as e:
            print(f"Error searching DifferentiableMemory ({memory_component.__class__.__name__}): {e}")
        return retrieved_items

    def _search_persistent_mem(
        self, 
        query_emb_single: torch.Tensor, 
        k_to_retrieve: int,
        similarity_exponent: float = 1.0
    ) -> List[Tuple[float, torch.Tensor]]:
        """Helper pour rechercher dans VectorMemoryStore (Persistent Memory)."""
        retrieved_items: List[Tuple[float, torch.Tensor]] = []
        if self.persistent_memory.active_entries == 0:
            return retrieved_items

        try:
            indices, similarities = self.persistent_memory.search(query_emb_single, top_k=k_to_retrieve)
            
            if indices is not None and similarities is not None:
                if similarity_exponent != 1.0 and similarities.numel() > 0:
                    similarities = torch.clamp(similarities, 0.0, 1.0) ** similarity_exponent

                if k_to_retrieve > 0 and indices.numel() > 0:
                    # persistent_memory.search peut déjà appliquer top_k, mais on re-vérifie
                    # Les similarités et indices sont [1, actual_k_retrieved]
                    num_retrieved = indices.size(1)
                    actual_k = min(k_to_retrieve, num_retrieved)
                    
                    # Si un exposant a été appliqué, nous pourrions avoir besoin de re-trier
                    if similarity_exponent != 1.0 and num_retrieved > 0:
                        top_scores, top_indices_in_retrieved = torch.topk(similarities.squeeze(0), actual_k)
                        original_indices = indices.squeeze(0)[top_indices_in_retrieved]
                    elif num_retrieved > 0:
                        top_scores = similarities.squeeze(0)[:actual_k]
                        original_indices = indices.squeeze(0)[:actual_k]
                    else:
                        return retrieved_items

                    for i in range(actual_k):
                        idx = original_indices[i].item()
                        score = top_scores[i].item()
                        retrieved_items.append((score, self.persistent_memory.memory_values[idx]))
        except Exception as e:
            print(f"Error searching PersistentMemory: {e}")
        return retrieved_items
        
    def _attention_retrieval(
        self,
        memory_component: Union['DifferentiableMemory', 'VectorMemoryStore'],
        query_emb: torch.Tensor,
        k_to_retrieve: int
    ) -> List[Tuple[float, torch.Tensor]]:
        """
        Effectue une récupération en utilisant un mécanisme d'attention.
        Retourne les 'top_k' éléments les plus pertinents basés sur les scores d'attention.
        """
        if not hasattr(self, 'retrieval_attention'):
            raise RuntimeError("La stratégie 'attention_retrieval' a été sélectionnée, mais le module retrieval_attention n'est pas initialisé.")

        if isinstance(memory_component, DifferentiableMemory):
            if not memory_component.initialized: return []
            mem_keys = memory_component.memory_keys[0] # [mem_size, key_dim]
            mem_values = memory_component.memory_values[0] # [mem_size, value_dim]
        elif isinstance(memory_component, VectorMemoryStore):
            if memory_component.active_entries == 0: return []
            # On utilise toutes les entrées actives
            mem_keys = memory_component.memory_keys[:memory_component.active_entries]
            mem_values = memory_component.memory_values[:memory_component.active_entries]
        else:
            return []

        if mem_keys.nelement() == 0:
            return []
            
        # Reshape pour MultiheadAttention: (Batch, Seq, Dim)
        # Ici, Batch=1, Seq=1 pour la requête, et Seq=mem_size pour les clés/valeurs
        query = query_emb.unsqueeze(0) # [1, 1, hidden_size]
        keys = mem_keys.unsqueeze(0)   # [1, mem_size, key_size]
        values = mem_values.unsqueeze(0) # [1, mem_size, value_size]

        # Le module d'attention retourne le contexte et les poids
        # Ici, le contexte n'est pas directement le résultat, ce sont les poids qui nous intéressent.
        _, attn_weights = self.retrieval_attention(query, keys, values)
        # attn_weights: [1, 1, mem_size]
        
        attn_scores = attn_weights.squeeze(0).squeeze(0) # [mem_size]

        k_to_retrieve = min(k_to_retrieve, attn_scores.size(0))
        top_scores, top_indices = torch.topk(attn_scores, k_to_retrieve)

        retrieved_items = []
        for score, idx in zip(top_scores, top_indices):
            retrieved_items.append((score.item(), mem_values[idx]))
            
        return retrieved_items

    def search(self, query_embedding: torch.Tensor, k: Optional[int] = None) -> Dict[str, List[Tuple[float, torch.Tensor]]]:
        """
        Recherche dans tous les niveaux de la mémoire en utilisant la stratégie configurée.

        Args:
            query_embedding: Le tenseur de la requête (batch_size=1, seq_len=1, hidden_size).
            k: Le nombre d'éléments à récupérer (optionnel, utilise la config par défaut).

        Returns:
            Un dictionnaire contenant les résultats de recherche pour chaque type de mémoire.
        """
        if query_embedding.dim() != 3 or query_embedding.size(0) != 1 or query_embedding.size(1) != 1:
            raise ValueError("La recherche attend un query_embedding de shape [1, 1, hidden_size].")
        
        query_emb_single = query_embedding.squeeze(0) # [1, hidden_size]

        strategy = self.config.memory_config.retrieval_strategy
        k_stm = k or self.config.memory_config.k_top_stm
        k_ltm = k or self.config.long_term_memory_config.top_k  # Corrigé: k_top_ltm -> top_k
        k_pm = k or self.config.memory_config.k_top_pm
        similarity_exponent = self.config.memory_config.similarity_exponent

        search_results: Dict[str, List[Tuple[float, torch.Tensor]]] = {
            "short_term": [],
            "long_term": [],
            "persistent": []
        }

        if strategy == "attention_retrieval":
            search_results["short_term"] = self._attention_retrieval(self.short_term_memory, query_emb_single, k_stm)
            search_results["long_term"] = self._attention_retrieval(self.long_term_memory, query_emb_single, k_ltm)
            search_results["persistent"] = self._attention_retrieval(self.persistent_memory, query_emb_single, k_pm)
        elif strategy == "cosine_similarity":
            search_results["short_term"] = self._search_differentiable_mem(self.short_term_memory, query_emb_single, k_stm, similarity_exponent)
            search_results["long_term"] = self._search_differentiable_mem(self.long_term_memory, query_emb_single, k_ltm, similarity_exponent)
            search_results["persistent"] = self._search_persistent_mem(query_emb_single, k_pm, similarity_exponent)
        else:
            raise ValueError(f"Stratégie de recherche inconnue : {strategy}")

        return search_results
        
    def _calculate_novelty_score(self, input_keys: torch.Tensor, memory_keys: torch.Tensor) -> float:
        """
        Calcule un score de nouveauté basé sur la similarité cosinus maximale entre
        les clés d'entrée et les clés de la mémoire.
        
        Args:
            input_keys: Clés de l'entrée actuelle [seq_len, key_size]
            memory_keys: Clés de la mémoire [mem_size, key_size]
            
        Returns:
            Un score de nouveauté (1 - similarité max).
        """
        if memory_keys is None or memory_keys.nelement() == 0 or input_keys.nelement() == 0:
            return 1.0 # Totalement nouveau si la mémoire est vide
            
        # Aplatir les dimensions batch/séquence pour la comparaison
        input_keys_flat = input_keys.reshape(-1, input_keys.size(-1))
        
        # Calculer la similarité cosinus
        # input_keys_flat: [N, key_size], memory_keys: [M, key_size]
        # sim_matrix: [N, M]
        sim_matrix = F.cosine_similarity(input_keys_flat.unsqueeze(1), memory_keys.unsqueeze(0), dim=2)
        
        # Le score de nouveauté est 1 moins la similarité maximale trouvée
        # Cela signifie qu'un score élevé indique que l'entrée est très différente de tout ce qui est en mémoire.
        max_similarity = torch.max(sim_matrix).item()
        
        return 1.0 - max_similarity

    def forward(
        self, 
        hidden_states: torch.Tensor,
        update_stm: bool = True,
        update_ltm: bool = True,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Passe avant pour la mémoire hiérarchique.
        Traite l'entrée, interagit avec les trois niveaux de mémoire,
        et retourne la sortie enrichie.
        
        Args:
            hidden_states: Représentations d'entrée [batch_size, seq_len, hidden_size]
            update_stm: Si True, met à jour la mémoire à court terme.
            update_ltm: Si True, met à jour la mémoire à long terme et persistante.
            attention_mask: Masque pour ignorer les éléments paddés.
            
        Returns:
            Tuple: (
                Tensor: Les états cachés de sortie, enrichis par la mémoire [batch, seq, hidden]
                Dict: Un dictionnaire contenant les sorties de chaque composant mémoire et des métadonnées.
            )
        """
        
        # Diviser en chunks si la séquence est trop longue
        max_chunk_size = getattr(self.config.memory_config, 'memory_chunk_size', 512)
        if hidden_states.size(1) > max_chunk_size:
            # Traitement par chunk
            outputs = []
            memory_outputs = []
            for i in range(0, hidden_states.size(1), max_chunk_size):
                chunk = hidden_states[:, i:i+max_chunk_size, :]
                mask_chunk = attention_mask[:, i:i+max_chunk_size] if attention_mask is not None else None
                
                output_chunk, mem_out_chunk = self._process_chunk(
                    chunk, 
                    update_stm=update_stm, 
                    update_ltm=update_ltm,
                    attention_mask=mask_chunk
                )
                outputs.append(output_chunk)
                memory_outputs.append(mem_out_chunk)
                
            final_output = torch.cat(outputs, dim=1)
            # Agréger les sorties de mémoire (ici, on prend la dernière pour simplifier)
            final_memory_output = memory_outputs[-1] if memory_outputs else {}
            
            return final_output, final_memory_output
        else:
            # Traitement direct
            return self._process_chunk(
                hidden_states, 
                update_stm=update_stm,
                update_ltm=update_ltm,
                attention_mask=attention_mask
            )
    
    def _process_chunk(
        self, 
        hidden_states: torch.Tensor,
        update_stm: bool = True,
        update_ltm: bool = True,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Traite un "chunk" de hidden_states à travers la mémoire.
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Projeter les hidden_states pour obtenir les requêtes de mémoire
        query_emb = self.query_projection(hidden_states) # [batch_size, seq_len, key_size]
        
        # Initialiser les listes pour stocker les résultats et les états de sortie
        output_hidden_states = torch.zeros_like(hidden_states)
        all_memory_outputs = {
            "stm_retrieved": [], "ltm_retrieved": [], "pm_retrieved": [],
            "stm_novelty": [], "ltm_novelty": [], "pm_novelty": [],
            "combined_retrieved": [], "gates": []
        }
        
        # Boucler sur chaque élément du batch, puis sur chaque pas de temps
        for b in range(batch_size):
            for i in range(seq_len):
                # Extraire la requête pour l'élément et le pas de temps courants
                # Le unsqueeze est crucial pour avoir le format [1, 1, dim] attendu par search
                current_query_emb = query_emb[b, i, :].unsqueeze(0).unsqueeze(0)
                current_hidden_state = hidden_states[b, i, :].unsqueeze(0).unsqueeze(0)
                
                # 1. Recherche dans les mémoires
                # La recherche est maintenant effectuée par élément, donc le batch_size est 1
                search_results = self.search(current_query_emb)
                stm_results = search_results["short_term"]
                ltm_results = search_results["long_term"]
                pm_results = search_results["persistent"]

                # Concaténer les résultats de toutes les mémoires
                retrieved_items = stm_results + ltm_results + pm_results
                all_memory_outputs["stm_retrieved"].append(stm_results)
                all_memory_outputs["ltm_retrieved"].append(ltm_results)
                all_memory_outputs["pm_retrieved"].append(pm_results)
                
                # ... (le reste de la logique de traitement pour un seul élément)
                # Cette partie combine les mémoires, met à jour les mémoires, etc.

                if retrieved_items:
                    # Empiler les valeurs récupérées pour former un tenseur
                    retrieved_values = torch.stack([item[1] for item in retrieved_items]).to(hidden_states.device)
                    retrieved_values = retrieved_values.unsqueeze(0) # [1, num_retrieved, value_dim]

                    # 2. Intégration de la mémoire via attention
                    # Le hidden state courant sert de requête, les mémoires récupérées servent de clé/valeur
                    # Note : On suppose value_dim == hidden_size pour l'attention
                    if retrieved_values.shape[2] != self.hidden_size:
                         # Si value_dim != hidden_size, il faudrait une projection.
                         # Pour l'instant, on saute l'attention si les dimensions ne correspondent pas.
                         # Ceci est une simplification. Une implémentation robuste nécessiterait une projection.
                         context_vector = self.output_projection(retrieved_values.mean(dim=1)) # Alternative simple
                    else:
                        context_vector, _ = self.memory_attention(
                            query=current_hidden_state, # [1, 1, hidden_size]
                            key=retrieved_values,       # [1, num_retrieved, hidden_size]
                            value=retrieved_values
                        ) # -> [1, 1, hidden_size]
                    
                    combined_output = context_vector.squeeze(1) # [1, hidden_size]
                    all_memory_outputs["combined_retrieved"].append(combined_output)
                else:
                    # S'il n'y a rien dans la mémoire, le contexte est nul
                    combined_output = torch.zeros_like(current_hidden_state).squeeze(1) # [1, hidden_size]
                    all_memory_outputs["combined_retrieved"].append(combined_output)

                # Mettre à jour l'état caché de sortie pour cet élément de la séquence
                output_hidden_states[b, i, :] = current_hidden_state.squeeze(0).squeeze(0) + combined_output.squeeze(0)

        # La mise à jour des mémoires STM/LTM/PM se ferait ici, après avoir traité tout le chunk.
        # Par exemple, en utilisant les `hidden_states` du chunk pour la mise à jour.
        # NOTE : La logique de mise à jour doit être revue pour être cohérente avec ce traitement par chunk.
        # Pour l'instant, on se concentre sur la correction de la recherche.
        # La logique de mise à jour ci-dessous est une simplification.
        
        current_keys = self.query_projection(hidden_states) # [batch_size, seq_len, key_size]

        if update_stm:
            self.short_term_memory(hidden_states, update_memory=True, keys=current_keys)

        if update_ltm:
            self.long_term_memory(hidden_states, update_memory=True, keys=current_keys)
        
        self.persistent_memory(hidden_states, update_memory=True) # PM gère ses clés en interne
        
        return output_hidden_states, all_memory_outputs
        
    def _efficient_attention(self, q, kv, mask=None):
        """
        Calcule une attention efficace pour de longues séquences.
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
