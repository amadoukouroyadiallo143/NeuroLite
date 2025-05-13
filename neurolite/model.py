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

from .config import NeuroLiteConfig
from .projection import MinHashBloomProjection, TokenizedMinHashProjection
from .mixer import MixerLayer, HyperMixer, FNetLayer
from .memory import DifferentiableMemory, ModernHopfieldLayer
from .routing import DynamicRoutingBlock
from .symbolic import NeuralSymbolicLayer, BayesianBeliefNetwork

# Imports des nouveaux modules AGI
from .multimodal import MultimodalProjection, CrossModalAttention
from .hierarchical_memory import HierarchicalMemory, VectorMemoryStore
from .continual import ContinualAdapter, ReplayBuffer, ProgressiveCompressor
from .reasoning import NeurosymbolicReasoner, StructuredPlanner


class NeuroLiteModel(nn.Module):
    """
    Modèle principal NeuroLite intégrant tous les composants.
    
    Cette architecture universelle et légère est conçue pour des applications mobiles
    et embarquées, fournissant une alternative efficace aux Transformers.
    Supporte diverses tâches comme la classification, la génération de texte et l'étiquetage de séquence.
    """
    
    def __init__(self, config: NeuroLiteConfig, task_type: str = "base", num_labels: int = None, tokenizer=None):
        super().__init__()
        
        self.config = config
        self.task_type = task_type  # Options: 'base', 'classification', 'sequence_labeling', 'generation'
        self.num_labels = num_labels
        self.tokenizer = tokenizer
        
        # Couche de projection d'entrée (remplace les embeddings traditionnels)
        if config.input_projection_type == "minhash_bloom":
            self.input_projection = MinHashBloomProjection(
                output_dim=config.hidden_size,
                minhash_permutations=config.minhash_num_permutations,
                bloom_filter_size=config.bloom_filter_size,
                dropout_rate=config.dropout_rate
            )
        else:
            # Version tokenisée pour travailler avec des indices de tokens
            self.input_projection = TokenizedMinHashProjection(
                output_dim=config.hidden_size,
                minhash_permutations=config.minhash_num_permutations,
                bloom_filter_size=config.bloom_filter_size,
                vocab_size=config.vocab_size,
                dropout_rate=config.dropout_rate
            )
            
        # Couches de traitement principales (MLP-Mixer ou variantes)
        self.layers = nn.ModuleList()
        for i in range(config.num_mixer_layers):
            # Alterner différents types de couches selon la position
            if i % 3 == 0 and config.use_dynamic_routing:
                # Couche avec routage dynamique (MoE)
                self.layers.append(
                    DynamicRoutingBlock(
                        input_size=config.hidden_size,
                        hidden_size=config.token_mixing_hidden_size,
                        num_experts=config.num_experts,
                        top_k=config.routing_top_k,
                        dropout_rate=config.dropout_rate,
                        activation=config.activation
                    )
                )
            elif i % 3 == 1:
                # Couche Mixer standard
                self.layers.append(
                    MixerLayer(
                        dim=config.hidden_size,
                        seq_len=config.max_seq_length,
                        token_mixing_hidden_dim=config.token_mixing_hidden_size,
                        channel_mixing_hidden_dim=config.channel_mixing_hidden_size,
                        dropout_rate=config.dropout_rate,
                        activation=config.activation,
                        layer_norm_eps=config.layer_norm_epsilon
                    )
                )
            else:
                # Couche HyperMixer pour mieux gérer les séquences variables
                self.layers.append(
                    HyperMixer(
                        dim=config.hidden_size,
                        max_seq_len=config.max_seq_length,
                        token_mixing_hidden_dim=config.token_mixing_hidden_size,
                        channel_mixing_hidden_dim=config.channel_mixing_hidden_size,
                        dropout_rate=config.dropout_rate,
                        activation=config.activation,
                        layer_norm_eps=config.layer_norm_epsilon
                    )
                )
        
        # Mémoire externe - version améliorée avec hiérarchie
        if config.use_external_memory:
            if getattr(config, 'use_hierarchical_memory', False):
                self.memory = HierarchicalMemory(
                    hidden_size=config.hidden_size,
                    short_term_size=getattr(config, 'short_term_memory_size', 64),
                    long_term_size=getattr(config, 'long_term_memory_size', config.memory_size),
                    persistent_size=getattr(config, 'persistent_memory_size', 512),
                    value_size=config.memory_dim
                )
            else:
                # Mémoire originale pour compatibilité
                self.memory = DifferentiableMemory(
                    hidden_size=config.hidden_size,
                    memory_size=config.memory_size,
                    value_size=config.memory_dim,
                    update_rate=config.memory_update_rate
                )
        else:
            self.memory = None
            
        # Module symbolique et raisonnement (optionnels)
        if config.use_symbolic_module:
            if getattr(config, 'use_advanced_reasoning', False):
                self.symbolic = NeurosymbolicReasoner(
                    hidden_size=config.hidden_size,
                    symbolic_dim=getattr(config, 'symbolic_dim', 64),
                    num_inference_steps=getattr(config, 'num_inference_steps', 3),
                    dropout_rate=config.dropout_rate
                )
            else:
                # Module symbolique original pour compatibilité
                self.symbolic = NeuralSymbolicLayer(
                    hidden_size=config.hidden_size,
                    symbolic_rules_file=config.symbolic_rules_file,
                    dropout_rate=config.dropout_rate
                )
        else:
            self.symbolic = None
            
        # Module de planification (optionnel)
        if getattr(config, 'use_planning_module', False):
            self.planner = StructuredPlanner(
                hidden_size=config.hidden_size,
                num_planning_steps=getattr(config, 'num_planning_steps', 5),
                plan_dim=getattr(config, 'plan_dim', 64),
                dropout_rate=config.dropout_rate
            )
        else:
            self.planner = None
            
        # Couche de normalisation finale
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, 
            eps=config.layer_norm_epsilon
        )
        
        # Projection multimodale (optionnelle)
        if getattr(config, 'use_multimodal', False):
            self.input_projection = MultimodalProjection(
                output_dim=config.hidden_size,
                minhash_permutations=config.minhash_num_permutations,
                bloom_filter_size=config.bloom_filter_size,
                image_patch_size=getattr(config, 'image_patch_size', 16),
                dropout_rate=config.dropout_rate
            )
        
        # Adaptateur d'apprentissage continu (optionnel)
        if getattr(config, 'use_continual_learning', False):
            self.continual_adapter = ContinualAdapter(
                hidden_size=config.hidden_size,
                buffer_size=getattr(config, 'replay_buffer_size', 100),
                adaptation_rate=getattr(config, 'adaptation_rate', 0.1),
                drift_threshold=getattr(config, 'drift_threshold', 0.5),
                dropout_rate=config.dropout_rate
            )
        else:
            self.continual_adapter = None
            
        # Couches de sortie spécifiques selon le type de tâche
        if task_type == "classification" and num_labels is not None:
            # Couche de classification
            self.classifier = nn.Linear(config.hidden_size, num_labels)
        elif task_type == "sequence_labeling" and num_labels is not None:
            # Couche d'étiquetage de séquence
            self.classifier = nn.Linear(config.hidden_size, num_labels)
        elif task_type == "generation" and tokenizer is not None:
            # Couche de prédiction pour la génération de texte
            self.lm_head = nn.Linear(config.hidden_size, len(tokenizer.word_to_idx) if hasattr(tokenizer, 'word_to_idx') else config.vocab_size)
        else:
            # Couche de projection générique pour le modèle de base
            self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)
        
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
        input_texts: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        update_memory: bool = True,
        output_hidden_states: bool = False,
        multimodal_inputs: Optional[Dict[str, torch.Tensor]] = None,
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
            input_texts: Liste de textes d'entrée (si utilisation de MinHashBloomProjection)
            input_ids: Tensor d'indices de tokens (si utilisation de TokenizedMinHashProjection)
            attention_mask: Masque d'attention (optionnel)
            update_memory: Si True, met à jour la mémoire externe
            output_hidden_states: Si True, retourne aussi les états intermédiaires
            
        Returns:
            Tensor de sortie [batch_size, seq_length, hidden_size] ou
            Tuple de (sortie, liste d'états cachés) si output_hidden_states=True
        """
        # Vérifier que l'entrée est valide
        if input_texts is None and input_ids is None and multimodal_inputs is None:
            raise ValueError("Vous devez fournir au moins un type d'entrée: 'input_texts', 'input_ids' ou 'multimodal_inputs'")
            
        if input_texts is not None and input_ids is not None:
            raise ValueError("Veuillez fournir soit 'input_texts', soit 'input_ids', pas les deux")
        
        # Traiter l'entrée selon son type
        if input_texts is not None:
            hidden_states = self._process_text_input(input_texts)
        elif input_ids is not None:
            hidden_states = self._process_token_input(input_ids, attention_mask)
        elif multimodal_inputs is not None and hasattr(self.input_projection, 'process_multimodal'):
            hidden_states = self.input_projection.process_multimodal(multimodal_inputs)
        else:
            raise ValueError("Vous devez fournir soit 'input_texts', 'input_ids' ou 'multimodal_inputs'")
        
        # Stocker les états cachés intermédiaires si demandé
        all_hidden_states = [] if output_hidden_states else None
        
        # Initialiser les variables de sortie symboliques à None
        symbolic_outputs = None
        plan_outputs = None
        
        # Apprentissage continu (adaptation aux nouvelles distributions)
        if continuous_learning and hasattr(self, 'continual_adapter'):
            hidden_states = self.continual_adapter(hidden_states, update_memory=update_memory)
        
        # Passage à travers les couches principales
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            
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
                
        # Module de planification (si activé et requis)
        if self.planner is not None and use_planning:
            if return_symbolic:
                hidden_states, plan_outputs = self.planner(hidden_states, 
                                                         constraints=constraints,
                                                         return_plan=True)
            else:
                hidden_states = self.planner(hidden_states, constraints=constraints)
                
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
            output = self.output_projection(hidden_states)
            outputs["hidden_states"] = output
        
        # Ajouter les états cachés et les sorties symboliques si demandé
        if output_hidden_states:
            outputs["all_hidden_states"] = all_hidden_states
        if symbolic_outputs is not None and return_symbolic:
            outputs["symbolic_outputs"] = symbolic_outputs
        if plan_outputs is not None and use_planning:
            outputs["plan_outputs"] = plan_outputs
        
        # Retourner les sorties dans le format approprié
        if not return_dict:
            if self.task_type == "base":
                if output_hidden_states:
                    return outputs["hidden_states"], outputs["all_hidden_states"]
                return outputs["hidden_states"]
            else:
                return (loss, outputs["logits"]) if loss is not None else outputs["logits"]
        
        return outputs

    def save_pretrained(self, save_dir: str):
        """
        Sauvegarde le modèle dans un répertoire.
        
        Args:
            save_dir: Chemin du répertoire où sauvegarder le modèle
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Sauvegarder la configuration
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.__dict__, f, indent=4)
        
        # Sauvegarder les poids du modèle
        model_path = os.path.join(save_dir, "model.pt")
        torch.save(self.state_dict(), model_path)
        
        # Sauvegarder la mémoire persistante si disponible
        if hasattr(self, 'memory') and self.memory is not None:
            if hasattr(self.memory, 'save_persistent_memory'):
                memory_path = os.path.join(save_dir, "persistent_memory.pt")
                self.memory.save_persistent_memory(memory_path)
                print(f"Mémoire persistante sauvegardée dans {memory_path}")
        
        print(f"Modèle sauvegardé dans {save_dir}")

    @classmethod
    def from_pretrained(cls, model_dir: str, task_type: str = None, num_labels: int = None, tokenizer=None) -> "NeuroLiteModel":
        """
        Charge un modèle pré-entraîné depuis un répertoire.
        
        Args:
            model_dir: Répertoire contenant le modèle sauvegardé
            task_type: Type de tâche ('base', 'classification', 'sequence_labeling', 'generation')
            num_labels: Nombre de classes (pour classification ou étiquetage de séquence)
            tokenizer: Tokenizer pour la génération de texte
            
        Returns:
            Modèle NeuroLite chargé
        """
        # Charger la configuration
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        
        config = NeuroLiteConfig(**config_dict)
        
        # Instancier le modèle avec la configuration
        model = cls(config)
        
        # Charger les poids du modèle
        model_path = os.path.join(model_dir, "model.pt")
        if not os.path.exists(model_path):
            # Rétrocompatibilité avec l'ancien format
            model_path = os.path.join(model_dir, "pytorch_model.bin")
            
        model.load_state_dict(torch.load(model_path))
        
        # Charger la mémoire persistante si disponible
        memory_path = os.path.join(model_dir, "persistent_memory.pt")
        if os.path.exists(memory_path) and hasattr(model, 'memory') and model.memory is not None:
            if hasattr(model.memory, 'load_persistent_memory'):
                model.memory.load_persistent_memory(memory_path)
                print(f"Mémoire persistante chargée depuis {memory_path}")
        
        return model

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        pad_token_id: int = 0,
        eos_token_id: int = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Génère du texte à partir des indices d'entrée.
        
        Args:
            input_ids: Tensor d'indices de tokens [batch_size, seq_length]
            max_length: Longueur maximale de la séquence générée
            temperature: Température pour l'échantillonnage (plus élevée = plus aléatoire)
            top_k: Nombre de tokens les plus probables à considérer pour l'échantillonnage
            top_p: Probabilité cumulative pour l'échantillonnage nucleus
            do_sample: Si True, échantillonne selon les probabilités, sinon prend le token le plus probable
            num_beams: Nombre de faisceaux pour beam search (1 = pas de beam search)
            repetition_penalty: Pénalité pour la répétition des mêmes tokens
            pad_token_id: ID du token de padding
            eos_token_id: ID du token de fin de séquence
            attention_mask: Masque d'attention pour l'entrée
            
        Returns:
            Tensor des indices générés [batch_size, max_length]
        """
        # Vérifier que le modèle est configuré pour la génération
        if self.task_type != "generation" or not hasattr(self, 'lm_head'):
            raise ValueError("Ce modèle n'est pas configuré pour la génération de texte. "
                             "Initialisez-le avec task_type='generation' et un tokenizer.")
        
        # Récupérer EOS token ID depuis le tokenizer si non spécifié
        if eos_token_id is None and self.tokenizer is not None and hasattr(self.tokenizer, 'special_tokens'):
            eos_token_id = self.tokenizer.special_tokens.get('<EOS>', None)
        
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Initialiser les séquences générées avec les entrées
        generated = input_ids.clone()
        
        # Pour le beam search
        if num_beams > 1:
            return self._generate_beam_search(input_ids, max_length, num_beams, 
                                            pad_token_id, eos_token_id, attention_mask)
            
        # Génération auto-régressive token par token
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Calculer les logits pour la position actuelle
                outputs = self.forward(input_ids=generated, attention_mask=attention_mask)
                logits = outputs["logits"]
                
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
                        [attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1
                    )
                
                # Arrêter si tous les batchs ont généré EOS ou atteint max_length
                if eos_token_id is not None and (next_token == eos_token_id).all():
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
