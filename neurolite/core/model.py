"""
Classe de modèle principal pour NeuroLite.
Intègre tous les composants (projection, mixer, mémoire, routage, symbolique)
dans une architecture unifiée et légère.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Union, Tuple, Any
import os
import json
import time

from ..Configs.config import (
    TrainingConfig, 
    ModelArchitectureConfig, 
    LoggingConfig, 
    TokenizerConfig,
    NeuroLiteConfig
)
from .projection import MinHashBloomProjection, TokenizedMinHashProjection
from .mixer import MixerLayer, HyperMixer, FNetLayer
from ..memory import DifferentiableMemory, ModernHopfieldLayer
from ..routers.routing import DynamicRoutingBlock
from ..symbolic import NeuralSymbolicLayer, BayesianBeliefNetwork

# Imports des nouveaux modules AGI
from ..multimodal.multimodal import MultimodalProjection, MultimodalGeneration, CrossModalAttention
from ..tokenization import NeuroLiteTokenizer
from ..memory.hierarchical_memory import HierarchicalMemory, VectorMemoryStore
from ..continual.continual import ContinualAdapter, ReplayBuffer, ProgressiveCompressor
from ..reasoning.reasoning import NeurosymbolicReasoner, StructuredPlanner


class NeuroLiteModel(nn.Module):
    """
    Modèle principal NeuroLite intégrant tous les composants.
    
    Cette architecture universelle et légère est conçue pour des applications mobiles
    et embarquées, fournissant une alternative efficace aux Transformers.
    Supporte diverses tâches comme la classification, la génération de texte et l'étiquetage de séquence.
    """
    
    def __init__(self, config: NeuroLiteConfig, task_type: str = "base", num_labels: int = None, tokenizer=None):
        super().__init__()
        
        # Configuration du modèle
        if not isinstance(config, NeuroLiteConfig):
            raise ValueError("La configuration doit être une instance de NeuroLiteConfig")
            
        self.config = config
        self.model_config = config.model_config
        self.task_type = task_type  # Options: 'base', 'classification', 'sequence_labeling', 'generation'
        self.num_labels = num_labels
        self.tokenizer = tokenizer
        
        # Tokenizer multimodal avancé (si spécifié)
        self.multimodal_tokenizer = None
        if tokenizer is not None and isinstance(tokenizer, NeuroLiteTokenizer):
            self.multimodal_tokenizer = tokenizer
        
        # Couche de projection d'entrée
        self.multimodal_to_hidden_proj = None
        if self.model_config.use_multimodal_input:
            multimodal_proj_output_dim = self.model_config.multimodal_output_dim \
                if self.model_config.multimodal_output_dim > 0 else self.model_config.hidden_size
            
            # Utilisation du nouveau MultimodalProjection avec encodeurs spécialisés
            self.input_projection = MultimodalProjection(
                output_dim=multimodal_proj_output_dim,
                hidden_dim=self.model_config.multimodal_hidden_dim,
                dropout_rate=self.model_config.dropout_rate,
                use_cross_attention=self.model_config.use_cross_modal_attention,
                num_attention_heads=self.model_config.cross_modal_num_heads,
                image_size=getattr(self.model_config, 'image_size', 224),
                patch_size=self.model_config.multimodal_image_patch_size,
                max_audio_length_ms=getattr(self.model_config, 'max_audio_length_ms', 30000),
                max_video_frames=self.model_config.multimodal_video_num_sampled_frames,
                max_graph_nodes=getattr(self.model_config, 'max_graph_nodes', 32)
            )
            
            # Module de génération multimodale
            self.multimodal_generation = MultimodalGeneration(
                input_dim=self.model_config.hidden_size,
                hidden_dim=self.model_config.multimodal_hidden_dim,
                dropout_rate=self.model_config.dropout_rate,
                vocab_size=self.model_config.vocab_size,
                image_size=getattr(self.model_config, 'image_size', 224),
                audio_sample_rate=getattr(self.model_config, 'audio_sample_rate', 16000),
                audio_length_ms=getattr(self.model_config, 'audio_length_ms', 30000),
                video_frames=self.model_config.multimodal_video_num_sampled_frames,
                max_graph_nodes=getattr(self.model_config, 'max_graph_nodes', 32)
            )
            
            if multimodal_proj_output_dim != self.model_config.hidden_size:
                self.multimodal_to_hidden_proj = nn.Linear(
                    multimodal_proj_output_dim, 
                    self.model_config.hidden_size
                )
        else:
            # Original text-specific input projection
            if self.model_config.input_projection_type == "minhash_bloom":
                self.input_projection = MinHashBloomProjection(
                    output_dim=self.model_config.hidden_size,
                    minhash_permutations=self.model_config.minhash_num_permutations,
                    bloom_filter_size=self.model_config.bloom_filter_size,
                    dropout_rate=self.model_config.dropout_rate
                )
            else:  # "ngram_hash" or tokenized
                self.input_projection = TokenizedMinHashProjection(
                    output_dim=self.model_config.hidden_size,
                    minhash_permutations=self.model_config.minhash_num_permutations,
                    dropout_rate=self.model_config.dropout_rate
                )
            
        # Initialisation des couches du modèle
        self.layer_norm = nn.LayerNorm(
            self.model_config.hidden_size, 
            eps=self.model_config.layer_norm_eps
        )
        self.dropout = nn.Dropout(self.model_config.dropout_rate)
        
        # Couches de mélange (Mixer)
        self.mixer_layers = nn.ModuleList([
            MixerLayer(
                dim=self.model_config.hidden_size,
                seq_len=self.model_config.max_seq_length,
                token_mixing_hidden_dim=self.model_config.token_mixing_hidden_size,
                channel_mixing_hidden_dim=self.model_config.channel_mixing_hidden_size,
                dropout_rate=self.model_config.dropout_rate,
                activation=self.model_config.activation,
                layer_norm_eps=self.model_config.layer_norm_epsilon
            ) for _ in range(self.model_config.num_hidden_layers)
        ])
        
        # Couches de traitement principales (MLP-Mixer ou variantes)
        self.layers = nn.ModuleList()
        for i in range(self.model_config.num_hidden_layers):
            # Alterner différents types de couches selon la position
            if i % 3 == 0 and self.model_config.use_dynamic_routing:
                # Couche avec routage dynamique (MoE)
                self.layers.append(
                    DynamicRoutingBlock(
                        input_size=self.model_config.hidden_size,
                        hidden_size=self.model_config.token_mixing_hidden_size,
                        num_experts=self.model_config.num_experts,
                        top_k=self.model_config.routing_top_k,
                        dropout_rate=self.model_config.dropout_rate,
                        activation=self.model_config.activation
                    )
                )
            elif i % 3 == 1:
                # Couche Mixer standard
                self.layers.append(
                    MixerLayer(
                        dim=self.model_config.hidden_size,
                        seq_len=self.model_config.max_seq_length,
                        token_mixing_hidden_dim=self.model_config.token_mixing_hidden_size,
                        channel_mixing_hidden_dim=self.model_config.channel_mixing_hidden_size,
                        dropout_rate=self.model_config.dropout_rate,
                        activation=self.model_config.activation,
                        layer_norm_eps=self.model_config.layer_norm_epsilon
                    )
                )
            else:
                # Couche HyperMixer pour mieux gérer les séquences variables
                self.layers.append(
                    HyperMixer(
                        dim=self.model_config.hidden_size,
                        max_seq_len=self.model_config.max_seq_length,
                        token_mixing_hidden_dim=self.model_config.token_mixing_hidden_size,
                        channel_mixing_hidden_dim=self.model_config.channel_mixing_hidden_size,
                        dropout_rate=self.model_config.dropout_rate,
                        activation=self.model_config.activation,
                        layer_norm_eps=self.model_config.layer_norm_epsilon
                    )
                )
        
        # Mémoire externe - version améliorée avec hiérarchie
        if self.model_config.use_external_memory:
            if getattr(self.model_config, 'use_hierarchical_memory', False):
                self.memory = HierarchicalMemory(
                    config=self.config,  # Pass the main NeuroLiteConfig object
                    hidden_size=self.model_config.hidden_size, # These can stay as they are specific to memory architecture
                    short_term_size=getattr(self.model_config, 'short_term_memory_size', 64),
                    long_term_size=getattr(self.model_config, 'long_term_memory_size', self.model_config.memory_size),
                    persistent_size=getattr(self.model_config, 'persistent_memory_size', 512),
                    value_size=self.model_config.memory_dim # This is likely memory_config.memory_dim or model_config.memory_dim
                    # HierarchicalMemory's __init__ will use config.memory_config for most of these anyway
                )
            else:
                # Mémoire originale pour compatibilité
                self.memory = DifferentiableMemory(
                    hidden_size=self.model_config.hidden_size,
                    memory_size=self.model_config.memory_size,
                    key_size=self.model_config.hidden_size,
                    value_size=self.model_config.memory_dim,
                    update_rate=self.model_config.memory_update_rate,
                    temperature=1.0  # Ajout de la température avec une valeur par défaut
                )
        else:
            self.memory = None
            
        # Module symbolique et raisonnement (optionnels)
        if self.model_config.use_symbolic_module:
            if getattr(self.model_config, 'use_advanced_reasoning', False):
                self.symbolic = NeurosymbolicReasoner(
                    hidden_size=self.model_config.hidden_size,
                    symbolic_dim=getattr(self.model_config, 'symbolic_dim', 64),
                    num_inference_steps=getattr(self.model_config, 'num_inference_steps', 3),
                    dropout_rate=self.model_config.dropout_rate,
                    memory_system=self.memory if isinstance(self.memory, HierarchicalMemory) else None
                )
            else:
                # Module symbolique original pour compatibilité
                self.symbolic = NeuralSymbolicLayer(
                    config=self.model_config,
                    hidden_size=self.model_config.hidden_size,
                    symbolic_rules_file=getattr(self.model_config, 'symbolic_rules_file', None),
                    dropout_rate=self.model_config.dropout_rate
                )
        else:
            self.symbolic = None

        # Réseau Bayésien (optionnel)
        if getattr(self.model_config, 'use_bayesian_module', False) and \
           getattr(self.model_config, 'num_bayesian_variables', 0) > 0:
            self.bayesian_network = BayesianBeliefNetwork(
                config=self.model_config,
                hidden_size=self.model_config.hidden_size,
                dropout_rate=self.model_config.dropout_rate
            )
        else:
            self.bayesian_network = None
            
        # Module de planification (optionnel)
        if getattr(self.model_config, 'use_planning_module', False):
            self.planner = StructuredPlanner(
                hidden_size=self.model_config.hidden_size,
                num_planning_steps=getattr(self.model_config, 'num_planning_steps', 5),
                plan_dim=getattr(self.model_config, 'plan_dim', 64),
                    dropout_rate=self.model_config.dropout_rate,
                    memory_system=self.memory if isinstance(self.memory, HierarchicalMemory) else None
            )
        else:
            self.planner = None
            
        # Couche de normalisation finale
        self.final_layer_norm = nn.LayerNorm(
            self.model_config.hidden_size, 
            eps=self.model_config.layer_norm_epsilon
        )
        
        # Projection multimodale (optionnelle) - This block is now handled above by replacing self.input_projection
        # if getattr(config, 'use_multimodal', False):
        #     self.input_projection = MultimodalProjection(
        #         output_dim=config.hidden_size,
        #         minhash_permutations=config.minhash_num_permutations,
        #         bloom_filter_size=config.bloom_filter_size,
        #         image_patch_size=getattr(config, 'image_patch_size', 16), # This was the old name
        #         dropout_rate=config.dropout_rate
        #     )
        
        # Adaptateur d'apprentissage continu (optionnel)
        if getattr(self.model_config, 'use_continual_adapter', False):
            self.continual_adapter = ContinualAdapter(
                hidden_size=self.model_config.hidden_size,
                buffer_size=getattr(self.model_config, 'continual_adapter_buffer_size', 100),
                adaptation_rate=getattr(self.model_config, 'continual_adapter_rate', 0.1),
                drift_threshold=getattr(self.model_config, 'continual_adapter_drift_threshold', 0.5),
                dropout_rate=self.model_config.dropout_rate,
                memory_system=self.memory if isinstance(self.memory, HierarchicalMemory) else None
            )
        else:
            self.continual_adapter = None

        # Compresseur progressif (optionnel)
        if getattr(self.model_config, 'use_progressive_compression', False):
            self.progressive_compressor = ProgressiveCompressor(
                hidden_size=self.model_config.hidden_size,
                compression_ratio=getattr(self.model_config, 'compression_ratio', 0.5)
            )
        else:
            self.progressive_compressor = None

        # Attention Cross-Modale (optionnelle)
        self.cross_modal_fuser = None
        if getattr(self.model_config, 'use_multimodal_input', False) and \
           getattr(self.model_config, 'use_cross_modal_attention', False):
            # This fuser will operate on model_config.hidden_size.
            # MultimodalProjection's output_dim might be different, requiring projection
            # handled in the forward pass if self.multimodal_to_hidden_proj exists.
            self.cross_modal_fuser = CrossModalAttention(
                hidden_size=self.model_config.hidden_size,
                num_heads=self.model_config.cross_modal_num_heads,
                dropout_rate=self.model_config.dropout_rate
            )
            
        # Couches de sortie spécifiques selon le type de tâche
        if task_type == "classification" and num_labels is not None:
            # Couche de classification
            self.classifier = nn.Linear(self.model_config.hidden_size, num_labels)
        elif task_type == "sequence_labeling" and num_labels is not None:
            # Couche d'étiquetage de séquence
            self.classifier = nn.Linear(self.model_config.hidden_size, num_labels)
        elif task_type == "generation":
            # Couche de prédiction pour la génération de texte
            vocab_size = self.model_config.vocab_size
            if tokenizer is not None:
                if hasattr(tokenizer, 'word_to_idx'):
                    vocab_size = len(tokenizer.word_to_idx)
                elif hasattr(tokenizer, 'vocab_size'):
                    vocab_size = tokenizer.vocab_size
            self.lm_head = nn.Linear(self.model_config.hidden_size, vocab_size)
        elif task_type == "multimodal_generation":
            # Cas spécial pour la génération multimodale (pas besoin de lm_head supplémentaire)
            # car le module multimodal_generation s'occupe déjà de la génération
            pass
        else:
            # Couche de projection générique pour le modèle de base
            self.output_projection = nn.Linear(
                self.model_config.hidden_size, 
                self.model_config.hidden_size
            )
        
    def _process_text_input(self, texts: List[str]) -> torch.Tensor:
        """
        Traite l'entrée texte brute avec la projection MinHash+Bloom.
        
        Args:
            texts: Liste de textes à traiter
            
        Returns:
            Tensor de représentations [batch_size, seq_length, hidden_size]
        """
        # Pour l'instant, nous considérons chaque texte comme un seul "token"
        # Dans une implémentation plus complète, il faudrait tokeniser le texte
        hidden_states = self.input_projection(texts)
        
        # Ajouter une dimension de séquence (seq_len=1 pour l'instant)
        hidden_states = hidden_states.unsqueeze(1)
        
        return hidden_states
        
    def _process_token_input(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Traite l'entrée tokenisée (indices de tokens).
        
        Args:
            input_ids: Tensor d'indices de tokens [batch_size, seq_length]
            attention_mask: Masque d'attention (optionnel) [batch_size, seq_length]
            
        Returns:
            Tensor de représentations [batch_size, seq_length, hidden_size]
        """
        # Projeter les indices de tokens en représentations
        hidden_states = self.input_projection(input_ids)
        
        # Appliquer le masque d'attention si fourni
        if attention_mask is not None:
            # Étendre le masque pour la dimension hidden_size
            mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            hidden_states = hidden_states * mask
            
        return hidden_states
    
    def forward(
        self,
        multimodal_inputs: Optional[Dict[str, Union[List[str], torch.Tensor, None]]] = None,
        attention_mask: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None,
        update_memory: bool = True,
        output_hidden_states: bool = False,
        external_facts: Optional[torch.Tensor] = None,
        use_planning: bool = False,
        constraints: Optional[torch.Tensor] = None,
        return_symbolic: bool = False,
        continuous_learning: bool = False,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]], Dict[str, Any]]:
        """
        Passage avant dans le modèle NeuroLite.
        
        Args:
            multimodal_inputs: Dictionnaire d'entrées multimodales.
                               Ex: {'text': ["texte"], 'image': tensor_image, 'audio': tensor_audio, 'video': tensor_video}
                               ou {'text': ["texte"]} ou {'input_ids': tensor_ids} pour le mode texte seul.
            attention_mask: Masque d'attention (principalement pour input_ids si utilisé en mode texte seul).
            update_memory: Si True, met à jour la mémoire externe et le buffer de l'adaptateur continu.
            output_hidden_states: Si True, retourne aussi les états intermédiaires.
            ... (autres args)
            
        Returns:
            Dictionnaire de sorties ou tenseur/tuple selon return_dict et task_type.
        """
        
        # Vérification des entrées
        if multimodal_inputs is None or not isinstance(multimodal_inputs, dict):
            raise ValueError("`multimodal_inputs` doit être un dictionnaire fourni.")
        
        # Utiliser le tokenizer multimodal si disponible
        if self.multimodal_tokenizer is not None and multimodal_inputs:
            # Tokenize les entrées multimodales
            tokenized_inputs = self.multimodal_tokenizer.tokenize(multimodal_inputs)
            # Mettre à jour multimodal_inputs avec les entrées tokenisées
            multimodal_inputs.update(tokenized_inputs)

        all_hidden_states = [] if output_hidden_states else None
        individual_modality_representations = None # To store outputs from MultimodalProjection

        if getattr(self.model_config, 'use_multimodal_input', False):
            if not isinstance(self.input_projection, MultimodalProjection):
                 raise TypeError("Modèle configuré pour entrée multimodale, mais self.input_projection n'est pas MultimodalProjection.")

            # Always request individual modalities if multimodal input is used.
            # The decision to use them for cross-attention is separate.
            fused_proj, individual_modality_representations = self.input_projection(
                multimodal_inputs,
                return_individual_modalities=True
            )

            # Determine the dimension of representations from input_projection's individual modalities
            # This is multimodal_proj_output_dim (if >0) or model_config.hidden_size
            # This is the dimension *before* potential projection by self.multimodal_to_hidden_proj
            current_repr_dim = self.model_config.multimodal_output_dim if \
                               self.model_config.multimodal_output_dim > 0 else \
                               self.model_config.hidden_size

            if getattr(self.model_config, 'use_cross_modal_attention', False) and \
               self.cross_modal_fuser is not None and \
               individual_modality_representations is not None and \
               len(individual_modality_representations) > 1:

                active_mod_reprs = [
                    rep.unsqueeze(1) for rep in individual_modality_representations.values() if rep is not None and torch.any(rep !=0)
                ] # List of [B, 1, current_repr_dim]

                if len(active_mod_reprs) > 1:
                    modality_sequence = torch.cat(active_mod_reprs, dim=1) # [B, num_modalities, current_repr_dim]

                    # Project if current_repr_dim is different from self.cross_modal_fuser's expected dim (model_config.hidden_size)
                    # This is handled by self.multimodal_to_hidden_proj if it exists
                    if self.multimodal_to_hidden_proj is not None:
                        # Reshape for Linear: [B * num_modalities, current_repr_dim]
                        batch_size_seq, num_modalities_seq, _ = modality_sequence.shape
                        modality_sequence_reshaped = modality_sequence.reshape(-1, current_repr_dim)
                        projected_sequence = self.multimodal_to_hidden_proj(modality_sequence_reshaped)
                        modality_sequence_projected = projected_sequence.reshape(batch_size_seq, num_modalities_seq, self.model_config.hidden_size)
                    elif current_repr_dim == self.model_config.hidden_size:
                        modality_sequence_projected = modality_sequence
                    else:
                        # Edge case: Dimensions mismatch, and no projection layer defined.
                        # Default to fused_proj, which should already be projected if needed by multimodal_to_hidden_proj
                        # This case should ideally not be hit if config is consistent.
                        hidden_states = fused_proj.unsqueeze(1)
                        if self.multimodal_to_hidden_proj is not None: # Should have been applied to fused_proj too
                             hidden_states = self.multimodal_to_hidden_proj(hidden_states)
                        # To prevent further execution in this branch for this scenario:
                        active_mod_reprs = [] # Force the else branch below for this specific sub-path

                    if len(active_mod_reprs) > 1: # Re-check after potential projection issue
                        fused_by_attention = self.cross_modal_fuser(
                            query_modality=modality_sequence_projected,
                            key_value_modality=modality_sequence_projected
                        ) # Output: [B, num_modalities, model_config.hidden_size]

                        hidden_states = fused_by_attention.mean(dim=1, keepdim=True) # [B, 1, model_config.hidden_size]
                    # else: already handled by the re-check logic for active_mod_reprs length

                else: # Only one or zero active modalities came from individual_reprs
                    hidden_states = fused_proj.unsqueeze(1)
                    if self.multimodal_to_hidden_proj is not None:
                         hidden_states = self.multimodal_to_hidden_proj(hidden_states)
            else: # Cross-modal attention (self.cross_modal_fuser) not used or not applicable
                hidden_states = fused_proj.unsqueeze(1)
                if self.multimodal_to_hidden_proj is not None:
                    hidden_states = self.multimodal_to_hidden_proj(hidden_states)
        else:
            # Mode texte seul: extraire input_texts ou input_ids de multimodal_inputs
            input_texts = multimodal_inputs.get("text")
            input_ids = multimodal_inputs.get("input_ids")
            # attention_mask est déjà un argument de la fonction forward

            if input_texts is not None and input_ids is not None:
                 raise ValueError("En mode texte seul, fournir soit 'text', soit 'input_ids' dans `multimodal_inputs`, pas les deux.")
            
            if input_texts is not None:
                if not isinstance(self.input_projection, MinHashBloomProjection):
                    raise TypeError("Type de projection d'entrée ne correspond pas à MinHashBloomProjection pour input_texts.")
                hidden_states = self._process_text_input(input_texts) # Attends une liste de textes
            elif input_ids is not None:
                if not isinstance(self.input_projection, TokenizedMinHashProjection):
                    raise TypeError("Type de projection d'entrée ne correspond pas à TokenizedMinHashProjection pour input_ids.")
                hidden_states = self._process_token_input(input_ids, attention_mask)
            else:
                raise ValueError("En mode texte seul, `multimodal_inputs` doit contenir 'text' (List[str]) ou 'input_ids' (Tensor).")

        if output_hidden_states:
            all_hidden_states.append(hidden_states.clone()) # Store initial projection
        
        # Initialiser les variables de sortie symboliques à None
        symbolic_outputs = None
        plan_outputs = None
        
        # Passage à travers les couches principales
        for i, layer_module in enumerate(self.layers): # Renamed 'layer' to 'layer_module' to avoid conflict
            hidden_states = layer_module(hidden_states)
            
            # The old cross-modal attention logic that was here (applied after layer i=0)
            # has been removed as the new self.cross_modal_fuser is applied at the input stage.
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
        
        # Intégration de la mémoire externe (si activée)
        if self.memory is not None:
            if isinstance(self.memory, HierarchicalMemory):
                hidden_states = self.memory(hidden_states, update_memory=update_memory, 
                                            attention_mask=attention_mask)
            else:
                hidden_states = self.memory(hidden_states, update_memory=update_memory)
            
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
                
        # Intégration du raisonnement symbolique (si activé)
        if self.symbolic is not None:
            if isinstance(self.symbolic, NeurosymbolicReasoner) and return_symbolic:
                hidden_states, symbolic_outputs = self.symbolic(hidden_states, 
                                                               external_facts=external_facts,
                                                               return_symbolic=True)
            else:
                hidden_states = self.symbolic(hidden_states)
            
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        # Intégration du Réseau Bayésien (si activé)
        if self.bayesian_network is not None:
            hidden_states = self.bayesian_network(hidden_states)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
                
        # Module de planification (si activé et requis)
        if self.planner is not None and use_planning:
            if return_symbolic: # This argument is 'return_symbolic', maybe should be 'return_plan_details'?
                hidden_states, plan_outputs = self.planner(hidden_states, 
                                                         constraints=constraints,
                                                         return_plan=True)
            else:
                hidden_states = self.planner(hidden_states, constraints=constraints)
                
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        # Adaptateur d'apprentissage continu (après les couches principales et autres modules)
        # Renamed 'continuous_learning' parameter to 'use_continual_adapter_during_forward' for clarity
        # Or, we can assume if self.continual_adapter exists, it's always used.
        # The original forward param was `continuous_learning`. Let's assume it controls this.
        if continuous_learning and self.continual_adapter is not None:
            hidden_states = self.continual_adapter(hidden_states, update_memory=update_memory)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
        
        # Couche de normalisation finale
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Traitement selon le type de tâche
        outputs = {}
        loss = None
        
        # Gérer les sorties selon le type de tâche
        if self.task_type == "classification":
            # Classification - moyenne sur la dimension de séquence
            sequence_output = torch.mean(hidden_states, dim=1)
            logits = self.classifier(sequence_output)
            outputs["logits"] = logits
            
            # Calculer la perte si les labels sont fournis
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs["loss"] = loss
                
        elif self.task_type == "sequence_labeling":
            # Étiquetage de séquence - prédire un label par position
            logits = self.classifier(hidden_states)
            outputs["logits"] = logits
            
            # Calculer la perte si les labels sont fournis
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                # Réorganiser pour CrossEntropyLoss: (batch_size * seq_length, num_labels)
                active_logits = logits.view(-1, self.num_labels)
                active_labels = labels.view(-1)
                loss = loss_fct(active_logits, active_labels)
                outputs["loss"] = loss
                
        elif self.task_type == "generation":
            # Génération de texte - prédire le prochain token
            logits = self.lm_head(hidden_states)
            outputs["logits"] = logits
            
            # Calculer la perte si les labels sont fournis
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                # Réorganiser pour CrossEntropyLoss: (batch_size * seq_length, vocab_size)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                outputs["loss"] = loss
        else:
            # Modèle de base - projection générique
            if hasattr(self, 'output_projection'):
                output = self.output_projection(hidden_states)
                outputs["hidden_states"] = output
            else:
                # Si output_projection n'existe pas, utiliser hidden_states directement
                outputs["hidden_states"] = hidden_states
        
        # Ajouter les états cachés et les sorties symboliques si demandé
        if output_hidden_states:
            outputs["all_hidden_states"] = all_hidden_states
        if symbolic_outputs is not None and return_symbolic: # from NeuroSymbolicReasoner
            outputs["symbolic_outputs"] = symbolic_outputs
        if plan_outputs is not None and use_planning: # from StructuredPlanner
            outputs["plan_outputs"] = plan_outputs
        
        # Add individual modality representations to output if requested and available
        if getattr(self.config, 'use_multimodal_input', False) and \
           getattr(self.config, 'use_cross_modal_attention', False) and \
           individual_modality_representations is not None and return_dict:
            outputs["individual_modality_representations"] = individual_modality_representations
            
        # Retourner les sorties dans le format approprié
        if not return_dict:
            if self.task_type == "base":
                if output_hidden_states:
                    return outputs["hidden_states"], outputs["all_hidden_states"]
                return outputs["hidden_states"]
            else:
                return (loss, outputs["logits"]) if loss is not None else outputs["logits"]
        
        return outputs

    def save_pretrained(self, save_dir: str, lightweight=False, disable_memory=False, disable_adapters=False):
        """
        Sauvegarde le modèle dans un répertoire avec des options pour gérer les structures complexes.
        
        Args:
            save_dir: Chemin du répertoire où sauvegarder le modèle
            lightweight: Si True, sauvegarde uniquement les poids essentiels du modèle
            disable_memory: Si True, désactive temporairement les composants de mémoire avant la sauvegarde
            disable_adapters: Si True, désactive temporairement les adaptateurs avant la sauvegarde
        """
        import sys
        import copy
        
        # Augmenter temporairement la limite de récursion pour gérer les structures complexes
        original_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(10000)  # Limite très élevée pour prendre en charge les structures imbriquées
        
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # Sauvegarder la configuration
            config_path = os.path.join(save_dir, "config.json")
            
            # Fonction pour sérialiser de manière sécurisée
            def safe_serialize(obj):
                if obj is None:
                    return None
                elif isinstance(obj, (str, int, float, bool)):
                    return obj
                elif isinstance(obj, (list, tuple)):
                    return [safe_serialize(x) for x in obj]
                elif isinstance(obj, dict):
                    return {str(k): safe_serialize(v) for k, v in obj.items()}
                elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
                    return safe_serialize(obj.to_dict())
                elif hasattr(obj, '__dict__'):
                    return safe_serialize(obj.__dict__)
                else:
                    return str(obj)
            
            try:
                # Essayer de sérialiser avec to_dict()
                if hasattr(self.config, 'to_dict') and callable(self.config.to_dict):
                    config_dict = self.config.to_dict()
                else:
                    config_dict = safe_serialize(self.config.__dict__)
                
                # Nettoyer le dictionnaire des valeurs non sérialisables
                cleaned_config = {}
                for k, v in config_dict.items():
                    try:
                        json.dumps({k: v})  # Tester la sérialisation
                        cleaned_config[k] = v
                    except (TypeError, OverflowError) as e:
                        print(f"Avertissement: Champ '{k}' non sérialisable, conversion en chaîne")
                        cleaned_config[k] = str(v)
                
                # Sauvegarder le fichier de configuration
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(cleaned_config, f, indent=4, ensure_ascii=False)
                    
            except Exception as e:
                print(f"Erreur critique lors de la sérialisation de la configuration: {e}")
                print("Tentative de sauvegarde des champs de base...")
                
                # Dernier recours : sauvegarder uniquement les attributs de base
                base_config = {}
                if hasattr(self.config, '__dict__'):
                    for k, v in self.config.__dict__.items():
                        if not k.startswith('_'):  # Exclure les attributs privés
                            try:
                                base_config[k] = safe_serialize(v)
                            except Exception as inner_e:
                                base_config[k] = f"[SerializationError] {str(inner_e)}"
                
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(base_config, f, indent=4, ensure_ascii=False)
            
            # Stratégie de sauvegarde en fonction des options
            if lightweight:
                # Mode léger: ne sauvegarde que les poids essentiels sans les structures complexes
                print("Sauvegarde en mode léger (uniquement poids essentiels)...")
                # Filtrer le state_dict pour ne garder que les poids des couches essentielles
                state_dict = {k: v for k, v in self.state_dict().items() 
                             if not any(x in k for x in ['buffer', 'memory.cache', 'adapter.store'])}
                torch.save(state_dict, os.path.join(save_dir, "model.pt"))
            else:
                # Sauvegarde complète avec gestion des modules complexes
                memory_state = None
                adapter_state = None
                
                # Désactiver temporairement les modules si demandé
                if disable_memory and hasattr(self, 'memory') and self.memory is not None:
                    memory_state = copy.deepcopy(self.memory)
                    if hasattr(self.memory, 'short_term_memory'):
                        self.memory.short_term_memory = None
                    if hasattr(self.memory, 'long_term_memory'):
                        self.memory.long_term_memory = None
                
                if disable_adapters and hasattr(self, 'continual_adapter') and self.continual_adapter is not None:
                    adapter_state = copy.deepcopy(self.continual_adapter)
                    if hasattr(self.continual_adapter, 'replay_buffer'):
                        self.continual_adapter.replay_buffer = None
                
                # Sauvegarder les poids du modèle
                model_path = os.path.join(save_dir, "model.pt")
                state_dict = self.state_dict()
                torch.save(state_dict, model_path)
                
                # Restaurer les modules désactivés
                if memory_state is not None:
                    self.memory = memory_state
                
                if adapter_state is not None:
                    self.continual_adapter = adapter_state
            
            # Sauvegarder la mémoire persistante séparément (si disponible et non désactivée)
            if not disable_memory and hasattr(self, 'memory') and self.memory is not None:
                if hasattr(self.memory, 'save_persistent_memory'):
                    try:
                        memory_path = os.path.join(save_dir, "persistent_memory.pt")
                        self.memory.save_persistent_memory(memory_path)
                        print(f"Mémoire persistante sauvegardée dans {memory_path}")
                    except Exception as e:
                        print(f"Attention: Impossible de sauvegarder la mémoire persistante: {e}")
            
            print(f"Modèle sauvegardé avec succès dans {save_dir}")
        
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du modèle: {e}")
            raise
        
        finally:
            # Rétablir la limite de récursion d'origine
            sys.setrecursionlimit(original_limit)
            
        # Sauvegarder la mémoire persistante séparément (si disponible et non désactivée)
        if not disable_memory and hasattr(self, 'memory') and self.memory is not None:
            if hasattr(self.memory, 'save_persistent_memory'):
                try:
                    memory_path = os.path.join(save_dir, "persistent_memory.pt")
                    self.memory.save_persistent_memory(memory_path)
                    print(f"Mémoire persistante sauvegardée dans {memory_path}")
                except Exception as e:
                    print(f"Attention: Impossible de sauvegarder la mémoire persistante: {e}")
        
        print(f"Modèle sauvegardé avec succès dans {save_dir}")

    def generate(
        self,
        multimodal_inputs: Optional[Dict[str, Union[List[str], torch.Tensor, None]]] = None,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        pad_token_id: int = 0,
        eos_token_id: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        target_modalities: Optional[List[str]] = None
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Génère des sorties (texte, image, audio, vidéo, graphe) à partir d'entrées.
        
        Args:
            multimodal_inputs: Dictionnaire d'entrées multimodales.
            max_length: Longueur maximale de la séquence générée (pour texte)
            temperature: Température pour l'échantillonnage (plus élevée = plus aléatoire)
            top_k: Nombre de tokens les plus probables à considérer pour l'échantillonnage (texte)
            top_p: Probabilité cumulative pour l'échantillonnage nucleus (texte)
            do_sample: Si True, échantillonne selon les probabilités, sinon prend le token le plus probable
            num_beams: Nombre de faisceaux pour beam search (1 = pas de beam search)
            repetition_penalty: Pénalité pour la répétition des mêmes tokens (texte)
            pad_token_id: ID du token de padding (texte)
            eos_token_id: ID du token de fin de séquence (texte)
            attention_mask: Masque d'attention pour l'entrée (texte)
            target_modalities: Liste des modalités à générer (pour génération multimodale)
            
        Returns:
            Pour texte: Tensor des indices générés [batch_size, max_length]
            Pour multimodal: Dictionnaire contenant les sorties générées par modalité
        """
        # Vérifier que le modèle est configuré pour la génération
        if self.task_type not in ["generation", "multimodal_generation"]:
            raise ValueError("Ce modèle n'est pas configuré pour la génération. "
                         "Initialisez-le avec task_type='generation' ou 'multimodal_generation' et un tokenizer.")
        
        # Vérifier que nous avons soit lm_head, soit multimodal_generation selon le type de tâche
        if self.task_type == "generation" and not hasattr(self, 'lm_head'):
            raise ValueError("Modèle de génération de texte non correctement initialisé. "
                         "Assurez-vous d'avoir un tokenizer et un vocab_size valides.")
        elif self.task_type == "multimodal_generation" and not hasattr(self, 'multimodal_generation'):
            raise ValueError("Modèle de génération multimodale non correctement initialisé. "
                         "Assurez-vous d'activer use_multimodal_input dans la configuration.")
        
        # Récupérer EOS token ID depuis le tokenizer si non spécifié
        if eos_token_id is None and self.tokenizer is not None and hasattr(self.tokenizer, 'special_tokens'):
            eos_token_id = self.tokenizer.special_tokens.get('<EOS>', None)
        
        # Vérifier si nous sommes en mode génération multimodale
        if self.task_type == "multimodal_generation" and multimodal_inputs is not None:
            # Vérifier si nous avons des modalités cibles spécifiques, sinon utiliser toutes les modalités disponibles
            if target_modalities is None:
                target_modalities = [k for k in multimodal_inputs.keys() if multimodal_inputs[k] is not None]
            
            # Vérifier que nous avons au moins une modalité cible valide
            if not target_modalities:
                raise ValueError("Aucune modalité cible valide spécifiée pour la génération multimodale.")
                
            # Encoder les entrées multimodales
            outputs = self.forward(multimodal_inputs=multimodal_inputs, output_hidden_states=False, return_dict=True)
            
            # Gérer différents types de retours de la méthode forward de manière plus robuste
            hidden_states = None
            
            # Essayer différentes façons d'extraire les hidden_states
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states[-1] if isinstance(outputs.hidden_states, (list, tuple)) else outputs.hidden_states
            elif isinstance(outputs, dict):
                if 'last_hidden_state' in outputs:
                    hidden_states = outputs['last_hidden_state']
                elif 'hidden_states' in outputs:
                    hidden_states = outputs['hidden_states'][-1] if isinstance(outputs['hidden_states'], (list, tuple)) else outputs['hidden_states']
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                # Si c'est un tuple, le premier élément est généralement les hidden_states
                hidden_states = outputs[0]
            
            # Vérifier que nous avons bien des hidden_states
            if hidden_states is None:
                raise ValueError("Impossible de récupérer les états cachés à partir des sorties du modèle. "
                               "Assurez-vous que le modèle renvoie bien les hidden_states dans son forward.")
            
            try:
                # Utiliser la méthode forward du module de génération multimodale
                # au lieu de la méthode generate qui n'existe pas
                return self.multimodal_generation(
                    latent=hidden_states.mean(dim=1),  # Moyenne sur la dimension de séquence
                    target_modalities=target_modalities,
                    temperature=temperature
                )
            except Exception as e:
                raise RuntimeError(f"Erreur lors de la génération multimodale: {str(e)}") from e
        # Génération de texte classique
        if multimodal_inputs is None:
            raise ValueError("multimodal_inputs doit être fourni pour la génération.")
        
        # Préparer un dictionnaire pour le forward
        inputs_for_forward = {"input_ids": None}
        if "text" in multimodal_inputs:
            inputs_for_forward["text"] = multimodal_inputs["text"]
        elif "input_ids" in multimodal_inputs:
            inputs_for_forward["input_ids"] = multimodal_inputs["input_ids"]
        
        # Encoder les entrées
        outputs = self.forward(
            multimodal_inputs=inputs_for_forward, 
            attention_mask=attention_mask,
            return_dict=True  # S'assurer d'avoir un dictionnaire en sortie
        )
        
        # Gérer différents types de sorties du modèle de manière robuste
        hidden_states = None
        logits = None
        
        # Essayer d'extraire les hidden_states ou logits selon la structure de sortie
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states'):
            hidden_states = outputs.hidden_states[-1] if isinstance(outputs.hidden_states, (list, tuple)) else outputs.hidden_states
        elif hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, dict):
            if 'last_hidden_state' in outputs:
                hidden_states = outputs['last_hidden_state']
            elif 'hidden_states' in outputs:
                hidden_states = outputs['hidden_states'][-1] if isinstance(outputs['hidden_states'], (list, tuple)) else outputs['hidden_states']
            elif 'logits' in outputs:
                logits = outputs['logits']
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            # Si c'est un tuple, le premier élément est généralement les hidden_states ou logits
            if torch.is_tensor(outputs[0]):
                if len(outputs[0].shape) >= 2:  # Vérifier si c'est bien un tenseur de séquence
                    hidden_states = outputs[0]
                else:
                    logits = outputs[0]
        
        # Si nous avons des logits, les utiliser directement
        if logits is not None:
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            input_ids = next_token
        # Sinon, utiliser les hidden_states avec la tête de langage
        elif hidden_states is not None:
            if hasattr(self, 'lm_head'):
                logits = self.lm_head(hidden_states[:, -1, :])  # Utiliser le dernier token
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
                input_ids = next_token
            else:
                raise ValueError("Le modèle n'a pas de tête de langage (lm_head) pour la génération.")
        else:
            raise ValueError("Impossible de récupérer les hidden_states ou logits pour la génération. "
                           "Assurez-vous que le modèle renvoie bien ces valeurs dans son forward.")
        
        # Vérification des entrées pour la génération de texte
        if input_ids is None:
            raise ValueError("input_ids ou multimodal_inputs avec 'text'/'input_ids' doit être fourni pour la génération de texte.")
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Longueurs des séquences d'entrée et sortie
        cur_len = input_ids.size(1)
        max_len = max_length
        
        # Valeur par défaut pour eos_token_id si non spécifié
        if eos_token_id is None:
            eos_token_id = -1  # Token non utilisé
        
        # Utiliser beam search si num_beams > 1
        if num_beams > 1:
            return self._generate_beam_search(
                input_ids, max_length, num_beams, pad_token_id, eos_token_id, attention_mask
            )
            
        # Initialiser le tenseur de sortie avec les input_ids initiaux
        generated = input_ids.clone()
        
        # Génération auto-régressive token par token
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Créer un dictionnaire d'entrées pour le forward
                current_inputs = {"input_ids": generated}
                
                # Calculer les logits pour la position actuelle
                outputs = self.forward(
                    multimodal_inputs=current_inputs,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                
                # Extraire les logits de la sortie
                if isinstance(outputs, dict):
                    if 'logits' in outputs:
                        logits = outputs['logits']
                    elif 'hidden_states' in outputs and hasattr(self, 'lm_head'):
                        logits = self.lm_head(outputs['hidden_states'])
                    else:
                        raise ValueError("Impossible de récupérer les logits pour la génération.")
                else:
                    # Si la sortie n'est pas un dictionnaire, on suppose que c'est directement les logits
                    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                    
                    # Si ce ne sont pas des logits mais des hidden_states, les projeter
                    if hasattr(self, 'lm_head') and logits.dim() == 3 and logits.size(-1) != self.lm_head.out_features:
                        logits = self.lm_head(logits)
                
                # Prendre les logits de la dernière position
                next_token_logits = logits[:, -1, :]
                
                # Appliquer la pénalité de répétition
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for previous_token in generated[i]:
                            # Si le token a déjà été généré, réduire sa probabilité
                            if previous_token.item() in [pad_token_id]:
                                continue
                            next_token_logits[i, previous_token.item()] /= repetition_penalty
                
                # Appliquer la température
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Masquer les tokens spéciaux sauf EOS
                if self.tokenizer is not None and hasattr(self.tokenizer, 'special_tokens'):
                    for special_id, special_token in enumerate(self.tokenizer.special_tokens.values()):
                        if special_token != eos_token_id:  # Ne pas masquer EOS
                            next_token_logits[:, special_token] = -float('inf')
                
                if do_sample:
                    # Appliquer top_k
                    if top_k > 0:
                        top_k_values, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                        # Créer un masque pour les tokens à conserver
                        filter_value = -float('inf')
                        min_values = top_k_values[:, -1].unsqueeze(-1)
                        next_token_logits = torch.where(
                            next_token_logits < min_values, 
                            torch.ones_like(next_token_logits) * filter_value,
                            next_token_logits
                        )
                    
                    # Appliquer top_p (nucleus sampling)
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Créer un masque pour les tokens à exclure
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Décaler le masque pour garder au moins un token
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                        sorted_indices_to_remove[:, 0] = 0  # Toujours garder le top token
                        
                        # Appliquer le masque aux indices triés
                        for batch_idx in range(batch_size):
                            indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                            next_token_logits[batch_idx, indices_to_remove] = -float('inf')
                    
                    # Échantillonner le prochain token
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Prendre le token le plus probable
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Ajouter le token généré à la séquence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Mettre à jour le masque d'attention si fourni
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, attention_mask.new_ones((batch_size, 1), device=attention_mask.device)], 
                        dim=-1
                    )
                
                # Arrêter si tous les batchs ont généré EOS ou atteint max_length
                if eos_token_id is not None:
                    # Vérifier si le dernier token généré est EOS dans tous les batchs
                    if (next_token == eos_token_id).all():
                        break
                    
                    # Vérifier si EOS apparaît quelque part dans la séquence générée
                    eos_in_batch = (generated == eos_token_id).any(dim=1)
                    if eos_in_batch.all():
                        break
        
        return generated
    
    def _generate_beam_search(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        num_beams: int,
        pad_token_id: int,
        eos_token_id: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Génère du texte en utilisant beam search.
        
        Implémentation simplifiée de beam search pour la génération de texte.
        """
        # Cette implémentation de beam search est simplifiée
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Initialiser les scores de beam
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        # Dupliquer l'input pour chaque beam
        beam_input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, -1).contiguous()
        beam_input_ids = beam_input_ids.view(batch_size * num_beams, -1)
        
        if attention_mask is not None:
            beam_attention_mask = attention_mask.unsqueeze(1).expand(batch_size, num_beams, -1).contiguous()
            beam_attention_mask = beam_attention_mask.view(batch_size * num_beams, -1)
        else:
            beam_attention_mask = None
        
        # Générer séquentiellement
        for step in range(max_length - input_ids.shape[1]):
            # Calculer les logits
            outputs = self.forward(input_ids=beam_input_ids, attention_mask=beam_attention_mask)
            logits = outputs["logits"]
            next_token_logits = logits[:, -1, :]
            
            # Calculer les scores des prochains tokens
            next_token_scores = torch.log_softmax(next_token_logits, dim=-1)
            next_token_scores = next_token_scores + beam_scores.view(-1, 1)
            
            # Reshape pour traiter par batch
            next_token_scores = next_token_scores.view(batch_size, num_beams * next_token_scores.shape[-1])
            
            # Obtenir les top-k prochains tokens et leur score
            topk_scores, topk_indices = torch.topk(next_token_scores, num_beams, dim=1)
            
            # Convertir les indices en nouveaux tokens et indices de beam
            new_tokens = topk_indices % next_token_logits.shape[-1]
            beam_indices = topk_indices // next_token_logits.shape[-1]
            
            # Construire les nouveaux beams
            new_beam_input_ids = []
            for batch_idx in range(batch_size):
                for beam_idx in range(num_beams):
                    token = new_tokens[batch_idx, beam_idx].unsqueeze(0)
                    beam_idx_in_batch = beam_indices[batch_idx, beam_idx]
                    
                    # Récupérer l'ID du beam correspondant
                    beam_id = batch_idx * num_beams + beam_idx_in_batch
                    
                    # Concaténer le nouveau token au beam existant
                    new_ids = torch.cat([beam_input_ids[beam_id], token.unsqueeze(0)], dim=1)
                    new_beam_input_ids.append(new_ids)
            
            # Mettre à jour les beams
            beam_input_ids = torch.cat(new_beam_input_ids, dim=0)
            beam_scores = topk_scores
            
            # Mettre à jour le masque d'attention si nécessaire
            if beam_attention_mask is not None:
                new_beam_attention_mask = torch.cat([
                    beam_attention_mask,
                    beam_attention_mask.new_ones((beam_attention_mask.shape[0], 1))
                ], dim=1)
                beam_attention_mask = new_beam_attention_mask
            
            # Vérifier si tous les beams ont généré EOS
            if eos_token_id is not None:
                eos_generated = (beam_input_ids[:, -1] == eos_token_id).view(batch_size, num_beams).all(dim=1)
                if eos_generated.all():
                    break
        
        # Sélectionner le meilleur beam pour chaque batch
        selected_ids = []
        for batch_idx in range(batch_size):
            start_idx = batch_idx * num_beams
            best_score_idx = beam_scores[batch_idx].argmax().item()
            best_beam_idx = start_idx + best_score_idx
            selected_ids.append(beam_input_ids[best_beam_idx].unsqueeze(0))
        
        return torch.cat(selected_ids, dim=0)
