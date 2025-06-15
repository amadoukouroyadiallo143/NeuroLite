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

from neurolite.Configs.config import (
    TrainingConfig, 
    ModelArchitectureConfig, 
    LoggingConfig, 
    TokenizerConfig,
    NeuroLiteConfig,
    ReasoningConfig
)
from neurolite.memory import DifferentiableMemory, ModernHopfieldLayer
from neurolite.routers.routing import DynamicRoutingBlock
from neurolite.symbolic import NeuralSymbolicLayer, BayesianBeliefNetwork
from neurolite.multimodal.multimodal import MultimodalProjection, MultimodalGeneration, CrossModalAttention
from neurolite.tokenization.tokenizer import NeuroLiteTokenizer
from neurolite.core.ssm import SSMLayer
from neurolite.core.mixer import MixerLayer, FNetLayer, HyperMixer
from neurolite.memory.hierarchical_memory import HierarchicalMemory, VectorMemoryStore
from neurolite.continual.continual import ContinualAdapter, ReplayBuffer, ProgressiveCompressor
from neurolite.reasoning.reasoning import NeurosymbolicReasoner, StructuredPlanner
from neurolite.continual.curriculum import CurriculumManager
from neurolite.meta.metacontroller import MetaController
from neurolite.reasoning.causal import CausalInferenceEngine, CausalGraph
from neurolite.tokenization.projectors import CrossModalProjector

class NeuroLiteModel(nn.Module):
    """
    Modèle principal NeuroLite intégrant tous les composants.
    
    Cette architecture universelle et légère est conçue pour des applications mobiles
    et embarquées, fournissant une alternative efficace aux Transformers.
    Supporte diverses tâches comme la classification, la génération de texte et l'étiquetage de séquence.
    """
    
    def __init__(self, config: NeuroLiteConfig, task_type: str = "base", num_labels: int = None, tokenizer: NeuroLiteTokenizer = None):
        super().__init__()
        
        # Configuration du modèle
        if not isinstance(config, NeuroLiteConfig):
            raise ValueError("La configuration doit être une instance de NeuroLiteConfig")
            
        self.config = config
        self.model_config = config.model_config
        self.reasoning_config = config.reasoning_config
        self.memory_config = config.memory_config
        self.task_type = task_type  # Options: 'base', 'classification', 'sequence_labeling', 'generation'
        self.num_labels = num_labels
        
        if tokenizer is None:
            raise ValueError("Un NeuroLiteTokenizer doit être fourni.")
        self.tokenizer = tokenizer
        
        # --- Modules de contrôle et de raisonnement ---
        self.metacontroller = None
        if self.model_config.use_metacontroller:
            self.metacontroller = MetaController(self, config)

        self.causal_engine = None
        if self.reasoning_config.use_causal_reasoning:
            self.causal_engine = CausalInferenceEngine(CausalGraph(variables=[], edges=[], var_dims={}))
            
        self.symbolic_module = None
        if self.reasoning_config.use_symbolic_module:
            self.symbolic_module = NeurosymbolicReasoner(
                hidden_size=self.model_config.hidden_size,
                symbolic_dim=self.reasoning_config.symbolic_dim,
                num_inference_steps=self.reasoning_config.num_inference_steps,
                max_facts=self.reasoning_config.max_facts,
                dropout_rate=self.model_config.dropout_rate
            )

        self.planner = None
        if self.reasoning_config.use_planning_module:
            self.planner = StructuredPlanner(
                hidden_size=self.model_config.hidden_size,
                num_planning_steps=self.reasoning_config.num_planning_steps,
                plan_dim=self.reasoning_config.plan_dim,
                dropout_rate=self.model_config.dropout_rate
            )

        # --- Module de mémoire unifié ---
        self.memory_system = None
        if self.model_config.use_hierarchical_memory:
            self.memory_system = HierarchicalMemory(config=config)
        elif self.memory_config.use_external_memory:
            self.memory_system = DifferentiableMemory(
                hidden_size=self.memory_config.memory_dim,
                memory_size=self.memory_config.memory_size,
                num_heads=self.memory_config.num_memory_heads,
            )
        
        # --- Modules d'apprentissage continu ---
        self.continual_adapter = None
        if self.model_config.use_continual_adapter:
            self.continual_adapter = ContinualAdapter(
                hidden_size=self.model_config.hidden_size,
                adapter_bottleneck_dim=self.model_config.adapter_bottleneck_dim,
                task_embedding_dim=self.model_config.task_embedding_dim
            )
        
        # --- Initialisation du coeur du modèle ---
        
        # Le tokenizer gère l'encodage et la projection multimodale.
        # Le modèle a besoin d'une couche d'embedding pour les tokens discrets générés.
        # ResidualVQ génère `num_quantizers` codes par pas de temps.
        # Nous utilisons une table d'embedding par quantificateur et nous additionnons les résultats.
        self.input_embeddings = nn.ModuleList([
            nn.Embedding(
                self.tokenizer.config.codebook_size, 
                    self.model_config.hidden_size
            ) for _ in range(self.tokenizer.config.num_quantizers)
        ])
            
        # Initialisation des couches du modèle
        self.layer_norm = nn.LayerNorm(
            self.model_config.hidden_size, 
            eps=self.model_config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(self.model_config.dropout_rate)
        
        # Couches de traitement principales (unifiées)
        self.layers = nn.ModuleList()
        for i in range(self.model_config.num_hidden_layers):
            is_ssm_layer = False
            if self.model_config.use_ssm_layers:
                if self.model_config.ssm_layer_indices is not None:
                    if i in self.model_config.ssm_layer_indices:
                        is_ssm_layer = True
                elif self.model_config.ssm_layer_frequency is not None and self.model_config.ssm_layer_frequency > 0:
                    if i % self.model_config.ssm_layer_frequency == 0: # Exemple: une couche SSM toutes les N couches
                        is_ssm_layer = True
            
            is_fnet_layer = False
            if not is_ssm_layer and self.model_config.use_fnet_layers:
                if self.model_config.fnet_layer_indices is not None:
                    if i in self.model_config.fnet_layer_indices:
                        is_fnet_layer = True
                elif self.model_config.fnet_layer_frequency is not None and self.model_config.fnet_layer_frequency > 0:
                    if i % self.model_config.fnet_layer_frequency == 0:
                        is_fnet_layer = True

            if is_ssm_layer:
                self.layers.append(
                    SSMLayer(
                        dim=self.model_config.hidden_size,
                        d_state=self.model_config.ssm_d_state,
                        d_conv=self.model_config.ssm_d_conv,
                        expand_factor=self.model_config.ssm_expand_factor,
                        # Les autres paramètres SSM (dt_rank, bias, etc.) sont pris par défaut dans SSMLayer
                        # ou pourraient être ajoutés à ModelArchitectureConfig si nécessaire pour plus de contrôle.
                        dropout=self.model_config.dropout_rate # Utiliser le dropout global du modèle
                    )
                )
            elif is_fnet_layer:
                self.layers.append(
                    FNetLayer(
                        dim=self.model_config.hidden_size,
                        dropout_rate=self.model_config.dropout_rate,
                        activation=self.model_config.activation,
                        layer_norm_eps=self.model_config.layer_norm_epsilon
                    )
                )
            # Alterner différents types de couches selon la position si ce n'est pas une couche SSM ou FNet
            elif i % 3 == 0 and self.model_config.use_dynamic_routing:
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

        # Compresseur progressif (optionnel)
        if getattr(self.model_config, 'use_progressive_compression', False):
            self.progressive_compressor = ProgressiveCompressor(
                hidden_size=self.model_config.hidden_size,
                compression_ratio=getattr(self.model_config, 'compression_ratio', 0.5)
            )
        else:
            self.progressive_compressor = None

        # Attention Cross-Modale (optionnelle)
        self.cross_modal_attention_text_image = None
        self.project_text_for_cm_attn = None  # Projection for text features before CM
        self.project_image_for_cm_attn = None  # Projection for image features before CM

        if getattr(self.model_config, 'use_multimodal_input', False) and \
           getattr(self.model_config, 'use_cross_modal_attention', False):
            
            # CrossModalAttention will operate on features of hidden_size
            cm_interaction_dim = self.model_config.hidden_size 
            
            self.cross_modal_attention_text_image = CrossModalAttention(
                hidden_size=cm_interaction_dim,
                num_heads=self.model_config.cross_modal_num_heads,
                dropout_rate=self.model_config.dropout_rate
            )
            
            # Determine if individual modality features need projection to cm_interaction_dim
            # This is the output dimension of MultimodalProjection's individual modality features
            multimodal_proj_individual_feature_dim = (
                self.model_config.multimodal_output_dim 
                if getattr(self.model_config, 'multimodal_output_dim', 0) > 0 
                else self.model_config.hidden_size
            )
            
            if multimodal_proj_individual_feature_dim != cm_interaction_dim:
                # These projections will be used if individual modality features (e.g., text, image)
                # from MultimodalProjection are not already of cm_interaction_dim (hidden_size).
                self.project_text_for_cm_attn = nn.Linear(multimodal_proj_individual_feature_dim, cm_interaction_dim)
                self.project_image_for_cm_attn = nn.Linear(multimodal_proj_individual_feature_dim, cm_interaction_dim)
            # If multimodal_proj_individual_feature_dim is already cm_interaction_dim (i.e., hidden_size),
            # then these projection layers will remain None, and original features will be used directly.
            
        # --- Têtes de sortie et génération ---
        # Le module MultimodalGeneration est maintenant la seule tête de sortie.
        # Il contient tous les décodeurs nécessaires pour toutes les tâches.
        self.multimodal_generation = MultimodalGeneration(self.model_config)
        
        # Les têtes de classification et de LM spécifiques sont supprimées.
        # La logique est maintenant gérée par les décodeurs respectifs.

        # Les modules suivants sont maintenant gérés par le NeuroLiteTokenizer.
        # Ils sont supprimés de l'initialisation du modèle principal.
        self.multimodal_projection = None
        self.cross_modal_fusion = None
        self.embedding = None # Remplacé par input_embeddings
        self.input_projection = None
        
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
        # Projeter les indices de tokens en représentations via la couche d'embedding
        hidden_states = self.embedding(input_ids)
        
        # Appliquer le masque d'attention si fourni
        if attention_mask is not None:
            # Étendre le masque pour la dimension hidden_size
            mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            hidden_states = hidden_states * mask
            
        return hidden_states
    
    def forward(
        self,
        multimodal_inputs: Optional[Dict[str, Any]] = None,
        input_indices: Optional[List[torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None,
        # Nouveaux arguments pour contrôler les modules
        update_memory: bool = False,
        use_reasoning: bool = False,
        return_symbolic: bool = False,
        continuous_learning: bool = False,
        return_hidden_states: bool = False,
        return_attentions: bool = False,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Dict[str, Any], tuple]:
        
        # 1. Obtenir les indices discrets du tokenizer si non fournis
        if input_indices is None:
            if multimodal_inputs is not None:
                tokenized_output = self.tokenizer.tokenize(multimodal_inputs)
                # La sortie est une liste de tenseurs, un par quantifieur
                input_indices = tokenized_output
        else:
                raise ValueError("Vous devez fournir soit `multimodal_inputs` soit `input_indices`.")
        
        # 2. Obtenir les embeddings d'entrée à partir des indices
        # On somme les embeddings de chaque quantifieur
        initial_embeddings = torch.zeros(
            input_indices[0].shape[0], 
            input_indices[0].shape[1], 
            self.model_config.hidden_size,
            device=input_indices[0].device
        )
        for i, emb_layer in enumerate(self.input_embeddings):
            initial_embeddings += emb_layer(input_indices[i])

        # 3. Normalisation et Dropout initiaux
        hidden_states = self.layer_norm(initial_embeddings)
        hidden_states = self.dropout(hidden_states)

        # Variables pour stocker les sorties intermédiaires
        all_hidden_states = () if return_hidden_states else None
        all_attentions = () if return_attentions else None # Supposant que des couches d'attention pourraient exister
        symbolic_output = None

        # 4. Boucle principale sur les couches du modèle
        for layer in self.layers:
            if return_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer(hidden_states)
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        # 5. Têtes de sortie spécifiques à la tâche
        loss = None
        logits = None
        
        if self.task_type == 'classification':
            logits = self.classifier(hidden_states[:, 0, :]) # Utilise le token [CLS] équivalent
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        elif self.task_type == 'sequence_labeling':
            logits = self.label_projector(hidden_states)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        elif self.task_type == "multimodal_generation":
            # En mode entraînement, on passe les labels au décodeur pour le calcul de la perte
            if labels is not None:
                decoder_output = self.multimodal_generation(
                    latent_representation=hidden_states,
                    targets=labels,
                    target_modality='text'
                )
                loss = decoder_output.get('loss')
                logits = decoder_output.get('logits')
            # En mode inférence, on ne fait rien ici, on retourne le `hidden_states` brut.
            # La méthode `generate` s'occupera du reste.

        # 6. Retourner la sortie
        if not return_dict:
            output_tuple = ()
            if logits is not None:
                output_tuple += (logits,)
            if hidden_states is not None:
                output_tuple += (hidden_states,)
            if all_attentions is not None:
                output_tuple += (all_attentions,)
            return output_tuple

        return {
            'loss': loss,
            'logits': logits,
            'last_hidden_state': hidden_states,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions,
            'symbolic_output': symbolic_output,
        }

    def save_pretrained(self, save_dir: str, lightweight=False, disable_memory=False, disable_adapters=False):
        """
        Sauvegarde le modèle et sa configuration dans un répertoire.
        
        Args:
            save_dir: Répertoire de destination.
            lightweight: Si True, ne sauvegarde que la configuration, pas les poids.
            disable_memory: Si True, ne sauvegarde pas la mémoire persistante.
            disable_adapters: Si True, ne sauvegarde pas les adaptateurs.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 1. Sauvegarder les poids du modèle (state_dict)
        if not lightweight:
            model_weights_path = os.path.join(save_dir, "pytorch_model.bin")
            torch.save(self.state_dict(), model_weights_path)
            print(f"Poids du modèle sauvegardés dans {model_weights_path}")

        # 2. Sauvegarder la configuration
        # On crée une version "propre" de la configuration pour éviter les problèmes de sérialisation.
        # On ne garde que les sous-configurations qui sont des dataclasses.
        clean_config = {}
        if self.config:
            for config_name, config_obj in self.config.__dict__.items():
                if hasattr(config_obj, '__dataclass_fields__'):
                    clean_config[config_name] = config_obj.to_dict()
                # On peut aussi ajouter des champs simples si nécessaire
                elif isinstance(config_obj, (str, int, float, bool, list, dict, type(None))):
                    clean_config[config_name] = config_obj

            config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(clean_config, f, indent=4)
        print(f"Configuration sauvegardée dans {config_path}")

        # 3. Sauvegarder le tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_dir)

        # 4. Sauvegarder la mémoire persistante
        if self.memory_system and hasattr(self.memory_system, 'save_persistent_memory') and not disable_memory:
            try:
                memory_path = os.path.join(save_dir, "persistent_memory")
                self.memory_system.save_persistent_memory(memory_path)
            except Exception as e:
                print(f"Attention: Impossible de sauvegarder la mémoire persistante: {e}")
                
        # 5. Sauvegarder les adaptateurs
        if self.continual_adapter and hasattr(self.continual_adapter, 'save_adapters') and not disable_adapters:
            try:
                adapter_path = os.path.join(save_dir, "adapters")
                self.continual_adapter.save_adapters(adapter_path)
                print(f"Adaptateurs sauvegardés avec succès dans {adapter_path}")
            except Exception as e:
                print(f"Attention: Impossible de sauvegarder les adaptateurs: {e}")
        
        print(f"Modèle sauvegardé avec succès dans {save_dir}")

    @classmethod
    def from_pretrained(cls, save_dir: str, task_type: str = "base", num_labels: int = None):
        """
        Charge un modèle et sa configuration depuis un répertoire.
        """
        if not os.path.isdir(save_dir):
            raise FileNotFoundError(f"Le répertoire du modèle spécifié n'existe pas : {save_dir}")

        # 1. Charger la configuration globale
        config_path = os.path.join(save_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Fichier de configuration 'config.json' non trouvé dans {save_dir}")
                
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = NeuroLiteConfig.from_dict(config_dict)
        
        # Le tokenizer est maintenant autonome mais a besoin de la config pour s'initialiser
        tokenizer = NeuroLiteTokenizer.from_pretrained(save_dir, neurolite_config=config)

        model = cls(config, task_type=task_type, num_labels=num_labels, tokenizer=tokenizer)
        
        # Charger les poids
        weights_path = os.path.join(save_dir, "pytorch_model.bin")
        if os.path.exists(weights_path):
            print(f"Chargement des poids depuis : {weights_path}")
            try:
                state_dict = torch.load(weights_path, map_location="cpu")
                
                # Gérer le cas où le state_dict a une clé 'model'
                if 'model' in state_dict:
                    state_dict = state_dict['model']

                # Filtrer les clés inattendues ou manquantes
                model_state_dict = model.state_dict()
                filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
                
                missing_keys = [k for k in model_state_dict.keys() if k not in filtered_state_dict]
                unexpected_keys = [k for k in state_dict.keys() if k not in model_state_dict]

                if missing_keys:
                    print(f"Attention: Clés manquantes dans le state_dict: {missing_keys}")
                if unexpected_keys:
                    print(f"Attention: Clés inattendues dans le state_dict (ignorées): {unexpected_keys}")

                model.load_state_dict(filtered_state_dict, strict=False)

            except Exception as e:
                print(f"Erreur lors du chargement du state_dict: {e}")
        else:
            print("Attention: Aucun fichier de poids 'pytorch_model.bin' trouvé.")
        
        return model

    @torch.no_grad()
    def generate(
        self,
        multimodal_inputs: Dict[str, Any],
        target_modalities: List[str],
        update_memory: bool = False,
        continuous_learning: bool = False,
        return_symbolic: bool = False,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_beams: int = 1,
        do_sample: bool = True
    ) -> Dict[str, Any]:
        """
        Génère une sortie multimodale à partir d'une ou plusieurs entrées.
        """
        self.eval() # S'assurer que le modèle est en mode évaluation
        device = next(self.parameters()).device

        # 1. Encodage des entrées pour obtenir la représentation latente initiale
        # On utilise la méthode `forward` en mode inférence (pas de labels)
        outputs = self.forward(
            multimodal_inputs=multimodal_inputs, 
            return_dict=True
        )

        if 'last_hidden_state' not in outputs or outputs['last_hidden_state'] is None:
            raise ValueError("La passe forward n'a pas retourné de 'last_hidden_state'. Vérifiez la logique de `NeuroLiteModel.forward`.")

        latent_representation = outputs['last_hidden_state']
        generated_outputs = {}

        # 2. Raisonnement neurosymbolique (si activé)
        # On vérifie si le module existe et si l'utilisateur a demandé une sortie symbolique.
        if self.symbolic_module is not None and return_symbolic:
            task_info = {'goal': 'generation', 'target_modalities': target_modalities}
            reasoning_input = {
                'latent': latent_representation, 
                'task_info': task_info,
                'return_symbolic': True
            }

            reasoning_output = self.symbolic_module(reasoning_input)
            latent_representation = reasoning_output.get('refined_latent', latent_representation)
            if 'symbolic_output' in reasoning_output:
                generated_outputs['symbolic'] = reasoning_output['symbolic_output']

        # 3. Mise à jour de la mémoire (si activée)
        # On vérifie si le module de mémoire existe et si l'utilisateur a demandé la mise à jour.
        if self.memory_system is not None and update_memory:
            self.memory_system.add(keys=latent_representation, values=latent_representation)
            
        # TODO: Implémenter la logique d'apprentissage continu
        if continuous_learning:
            # Cette logique est complexe et sort du cadre de `generate`.
            pass

        # 4. Génération multimodale déléguée
        # On boucle sur chaque modalité cible et on appelle le générateur correspondant.
        decoded_outputs = {}
        for modality in target_modalities:
            # La méthode `generate` de MultimodalGeneration génère pour une seule modalité à la fois
            # et retourne un dictionnaire {modalité: sortie}.
            output_modality = self.multimodal_generation.generate(
                latent_representation=latent_representation,
                target_modality=modality, # Passer la modalité unique
                tokenizer=self.tokenizer,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                num_beams=num_beams
            )
            decoded_outputs.update(output_modality)
        
        # Fusionner les sorties décodées avec d'autres sorties (ex: symboliques)
        generated_outputs.update(decoded_outputs)

        return generated_outputs

    def _filter_logits(self, logits, top_k=0, top_p=1.0):
        if top_k > 0:
            filter_value = -float('Inf')
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')
            
        return logits
    
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
        Implémentation simplifiée de la recherche par faisceau.
        NOTE: Cette méthode est un placeholder et nécessiterait une implémentation
        beaucoup plus robuste pour une utilisation en production, probablement
        en s'inspirant des implémentations de bibliothèques comme `transformers`.
        """
        batch_size = input_ids.shape[0]
        
        # Étendre les entrées pour chaque faisceau
        expanded_batch_size = batch_size * num_beams
        expanded_input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, -1).reshape(expanded_batch_size, -1)
        
        # Scores des faisceaux
        beam_scores = torch.zeros((batch_size, num_beams), device=input_ids.device)
        beam_scores[:, 1:] = -1e9 # Forcer le choix du premier faisceau au début
        
        # Boucle de génération
        for _ in range(max_length):
            # Obtenir les logits
            # Cette partie est complexe car elle dépend de l'état interne du modèle
            # et de la manière dont on gère les `past_key_values` pour l'efficacité.
            # Pour l'instant, on se contente de passer les `input_ids` complets à chaque fois.
            
            # TODO: Obtenir les logits du modèle
            # outputs = self.forward(input_ids=expanded_input_ids)
            # next_token_logits = outputs.logits[:, -1, :]
            
            # --- Placeholder ---
            # Simuler l'obtention des logits du vocabulaire
            vocab_size = self.config.model_config.vocab_size
            next_token_logits = torch.randn(expanded_batch_size, vocab_size, device=input_ids.device)
            # --- Fin Placeholder ---
            
            # Appliquer les scores de faisceau aux logits
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            next_token_scores = next_token_scores + beam_scores.view(-1, 1).expand_as(next_token_scores)
            
            # Reshape pour trouver les meilleurs tokens sur tous les faisceaux
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            
            # Obtenir les top-k scores et tokens
            best_scores, best_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
            
            # Logique pour sélectionner les prochains faisceaux
            # ... (cette partie est très complexe et omise pour la concision) ...
            
            # Condition d'arrêt
            # if (next_tokens == eos_token_id).all():
            #     break
            
        # Retourner la meilleure séquence (placeholder)
        return expanded_input_ids

    def _update_weights(self, module):
        """Initialise les poids du modèle."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if hasattr(module.weight, 'data'):
                module.weight.data.normal_(mean=0.0, std=self.config.model_config.initializer_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    @property
    def device(self):
        return next(self.parameters()).device
