"""
Configuration pour les modèles NeuroLite.

Ce module définit la configuration de base pour les modèles NeuroLite,
permettant une personnalisation fine de l'architecture et du comportement.
"""

"""
Configuration principale pour les modèles NeuroLite.

Ce module définit toutes les configurations nécessaires pour initialiser
et entraîner les modèles NeuroLite, y compris les configurations pour le tokenizer multimodal.
"""

from dataclasses import dataclass, field, fields
from typing import List, Optional, Dict, Any, Tuple, Union, Literal, TypeVar, Type
from pathlib import Path
import os
import json
import torch
import yaml
from copy import deepcopy

# Type variable for config classes
T = TypeVar('T', bound='BaseConfig')


@dataclass
class BaseConfig:
    """Classe de base pour toutes les configurations."""
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit la configuration en dictionnaire en gérant correctement les types de données.
        
        Returns:
            Dict[str, Any]: Dictionnaire contenant la configuration sérialisée
        """
        output = {}
        
        # Si c'est une dataclass, utiliser __dataclass_fields__
        if hasattr(self, '__dataclass_fields__'):
            fields = self.__dataclass_fields__
        # Sinon, utiliser __dict__ directement
        else:
            fields = self.__dict__.keys()
            
        for field_name in fields:
            try:
                # Récupérer la valeur du champ
                value = getattr(self, field_name)
                
                # Gérer les types de base (None, str, int, float, bool)
                if value is None or isinstance(value, (str, int, float, bool)):
                    output[field_name] = value
                    continue
                    
                # Gérer les listes et tuples
                if isinstance(value, (list, tuple)):
                    output[field_name] = [
                        v.to_dict() if hasattr(v, 'to_dict') 
                        else self._convert_to_serializable(v) 
                        for v in value
                    ]
                    continue
                    
                # Gérer les dictionnaires
                if isinstance(value, dict):
                    output[field_name] = {
                        k: v.to_dict() if hasattr(v, 'to_dict') 
                        else self._convert_to_serializable(v)
                        for k, v in value.items()
                    }
                    continue
                
                # Gérer les objets avec méthode to_dict
                if hasattr(value, 'to_dict'):
                    output[field_name] = value.to_dict()
                    continue
                    
                # Gérer les dataclasses
                if hasattr(value, '__dataclass_fields__'):
                    output[field_name] = self._convert_to_serializable(value)
                    continue
                    
                # Pour tous les autres types, convertir en chaîne
                output[field_name] = str(value) if value is not None else None
                
            except Exception as e:
                print(f"Avertissement: Impossible de sérialiser le champ {field_name}: {e}")
                output[field_name] = None
                
        return output
        
    def _convert_to_serializable(self, obj):
        """
        Convertit récursivement un objet en un format sérialisable.
        
        Args:
            obj: L'objet à convertir
            
        Returns:
            Une version sérialisable de l'objet
            
        Gère les types suivants:
        - Types de base (None, str, int, float, bool)
        - Listes, tuples, sets
        - Dictionnaires
        - Objets avec méthode to_dict()
        - Dataclasses
        - Objets PyTorch (Tensor, Module)
        - Objets Path
        - Objets Enum
        - Autres types (convertis en chaîne)
        """
        # Gérer les types de base
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
            
        # Gérer les listes, tuples et sets
        if isinstance(obj, (list, tuple, set)):
            return [self._convert_to_serializable(x) for x in obj]
            
        # Gérer les dictionnaires
        if isinstance(obj, dict):
            return {
                str(k): self._convert_to_serializable(v) 
                for k, v in obj.items()
                if isinstance(k, (str, int, float, bool))  # Clés doivent être des types de base
            }
            
        # Gérer les objets PyTorch
        if 'torch' in str(type(obj).__module__):
            if hasattr(obj, 'tolist'):  # Pour les tenseurs
                return obj.tolist()
            if hasattr(obj, 'state_dict'):  # Pour les modules PyTorch
                return {
                    'type': type(obj).__name__,
                    'state_dict': self._convert_to_serializable(obj.state_dict())
                }
            return str(obj)
            
        # Gérer les Path objects
        if hasattr(obj, '__fspath__'):
            return str(obj)
            
        # Gérer les Enums
        if hasattr(obj, 'name') and hasattr(obj, 'value') and type(obj).__name__ == 'Enum':
            return obj.name
            
        # Gérer les objets avec méthode to_dict
        if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            try:
                return self._convert_to_serializable(obj.to_dict())
            except Exception as e:
                print(f"Avertissement: Échec de to_dict() sur {type(obj).__name__}: {e}")
                # Continuer pour essayer d'autres méthodes
                
        # Gérer les dataclasses
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for field_name, field_info in obj.__dataclass_fields__.items():
                try:
                    # Ne pas sérialiser les champs qui commencent par _
                    if field_name.startswith('_'):
                        continue
                        
                    field_value = getattr(obj, field_name, None)
                    
                    # Si le champ a une valeur par défaut et que la valeur actuelle est égale à la valeur par défaut,
                    # on peut l'ignorer pour économiser de l'espace
                    if 'default' in field_info.metadata and field_value == field_info.metadata['default']:
                        continue
                        
                    result[field_name] = self._convert_to_serializable(field_value)
                except Exception as e:
                    print(f"Avertissement: Impossible de sérialiser le champ {field_name}: {e}")
                    result[field_name] = None
            return result
            
        # Gérer les objets avec __dict__
        if hasattr(obj, '__dict__'):
            return {
                k: self._convert_to_serializable(v)
                for k, v in obj.__dict__.items()
                if not k.startswith('_')
            }
            
        # Pour tous les autres types, essayer de les convertir en chaîne
        try:
            return str(obj)
        except Exception as e:
            print(f"Avertissement: Impossible de convertir l'objet de type {type(obj).__name__} en chaîne: {e}")
            return None
    
    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """Crée une configuration à partir d'un dictionnaire."""
        return cls(**config_dict)
    
    def update(self, **kwargs):
        """Met à jour les paramètres de configuration."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration pour l'entraînement du modèle.
    
    Attributes:
        train_data_path: Chemin vers les données d'entraînement
        val_data_path: Chemin vers les données de validation
        output_dir: Répertoire de sortie pour les checkpoints
        logging_dir: Répertoire pour les logs
        num_train_epochs: Nombre d'époques d'entraînement
        per_device_train_batch_size: Taille de batch par appareil
        per_device_eval_batch_size: Taille de batch pour l'évaluation
        warmup_steps: Nombre d'étapes de préchauffage
        weight_decay: Taux de décroissance des poids
        learning_rate: Taux d'apprentissage
        max_grad_norm: Valeur maximale pour le clipping du gradient
        gradient_accumulation_steps: Nombre d'étapes d'accumulation du gradient
        fp16: Activer l'entraînement en précision mixte FP16
        fp16_opt_level: Niveau d'optimisation pour FP16
        save_steps: Fréquence de sauvegarde des checkpoints
        eval_steps: Fréquence d'évaluation
        logging_steps: Fréquence de journalisation
        save_total_limit: Nombre maximum de checkpoints à conserver
        load_best_model_at_end: Charger le meilleur modèle à la fin de l'entraînement
        metric_for_best_model: Métrique pour sélectionner le meilleur modèle
        greater_is_better: Si la métrique doit être maximisée
    """
    # Chemins des données
    train_data_path: str = "data/processed/train"
    val_data_path: str = "data/processed/val"
    output_dir: str = "models/checkpoints"
    logging_dir: str = "logs"
    
    # Paramètres d'entraînement
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    learning_rate: float = 5e-5
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Précision mixte
    fp16: bool = True
    fp16_opt_level: str = "O1"
    
    # Fréquences
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    
    # Chargement du meilleur modèle
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Métriques à calculer
    metrics: List[str] = field(
        default_factory=lambda: ["accuracy", "f1", "precision", "recall"]
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire."""
        return {
            'train_data_path': self.train_data_path,
            'val_data_path': self.val_data_path,
            'output_dir': self.output_dir,
            'logging_dir': self.logging_dir,
            'num_train_epochs': self.num_train_epochs,
            'per_device_train_batch_size': self.per_device_train_batch_size,
            'per_device_eval_batch_size': self.per_device_eval_batch_size,
            'warmup_steps': self.warmup_steps,
            'weight_decay': self.weight_decay,
            'learning_rate': self.learning_rate,
            'max_grad_norm': self.max_grad_norm,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'fp16': self.fp16,
            'fp16_opt_level': self.fp16_opt_level,
            'save_steps': self.save_steps,
            'eval_steps': self.eval_steps,
            'logging_steps': self.logging_steps,
            'save_total_limit': self.save_total_limit,
            'load_best_model_at_end': self.load_best_model_at_end,
            'metric_for_best_model': self.metric_for_best_model,
            'greater_is_better': self.greater_is_better,
            'metrics': self.metrics
        }


@dataclass
class LoggingConfig(BaseConfig):
    """Configuration pour la journalisation et le suivi des expériences.
    
    Attributes:
        wandb_project: Nom du projet Weights & Biases
        wandb_run_name: Nom de l'exécution
        wandb_watch: Activer le suivi des poids avec W&B
        wandb_log_model: Télécharger le modèle final vers W&B
        log_level: Niveau de journalisation (debug, info, warning, error, critical)
    """
    wandb_project: str = "neurolite-training"
    wandb_run_name: str = "base-run"
    wandb_watch: bool = False
    wandb_log_model: bool = False
    log_level: str = "info"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire.
        
        Returns:
            Dict[str, Any]: Dictionnaire contenant la configuration sérialisée
        """
        output = {}
        for field_name in self.__dataclass_fields__:
            field_value = getattr(self, field_name, None)
            if field_value is not None:
                output[field_name] = self._convert_to_serializable(field_value)
        return output


@dataclass
class TokenizerEncoderConfig(BaseConfig):
    """Configuration de base pour les encodeurs de modalités."""
    hidden_size: int = 256
    num_layers: int = 3
    dropout_rate: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-6


@dataclass
class TextEncoderConfig(TokenizerEncoderConfig):
    """Configuration pour l'encodeur de texte."""
    vocab_size: int = 50000
    max_position_embeddings: int = 2048
    embedding_size: int = 256
    use_learned_position_embeddings: bool = True
    use_relative_position_bias: bool = True
    share_embeddings_with_lm_head: bool = True


@dataclass
class ImageEncoderConfig(TokenizerEncoderConfig):
    """Configuration pour l'encodeur d'image."""
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    use_adaptive_patches: bool = True
    use_patch_dropout: bool = True
    patch_dropout_rate: float = 0.1
    use_cls_token: bool = True


@dataclass
class AudioEncoderConfig(TokenizerEncoderConfig):
    """Configuration pour l'encodeur audio."""
    sampling_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    max_audio_length_ms: int = 30000
    use_spectrogram_augmentation: bool = True


@dataclass
class VideoEncoderConfig(TokenizerEncoderConfig):
    """Configuration pour l'encodeur vidéo."""
    num_frames: int = 8
    image_size: int = 224
    patch_size: int = 16
    temporal_patch_size: int = 2
    num_channels: int = 3
    use_temporal_attention: bool = True
    use_factorized_encoder: bool = True
    max_video_length_sec: float = 30.0


@dataclass
class GraphEncoderConfig(TokenizerEncoderConfig):
    """Configuration pour l'encodeur de graphe."""
    node_feature_dim: int = 128
    edge_feature_dim: int = 64
    num_node_types: int = 16
    num_edge_types: int = 8
    use_graph_attention: bool = True
    num_graph_attention_heads: int = 4
    max_num_nodes: int = 512
    max_num_edges: int = 2048


@dataclass
class QuantizerConfig(BaseConfig):
    """Configuration pour les quantificateurs vectoriels."""
    n_embeddings: int = 8192
    embedding_dim: int = 256
    commitment_cost: float = 0.25
    use_ema_updates: bool = True
    ema_decay: float = 0.99
    restart_unused_codes: bool = True
    threshold_ema_dead_code: float = 1e-5


@dataclass
class TokenizerConfig(BaseConfig):
    """Configuration principale pour le tokenizer multimodal NeuroLite."""
    # Paramètres généraux du tokenizer
    semantic_vocab_size: int = 8192
    detail_vocab_size: int = 32768
    hidden_size: int = 768
    dropout_rate: float = 0.1
    
    # Paramètres pour le codebook et la quantification vectorielle
    use_residual_vq: bool = True        # Utiliser la quantification vectorielle résiduelle
    num_residual_layers: int = 3        # Nombre de couches résiduelles pour la quantification
    use_dual_codebook: bool = True      # Utiliser un double codebook (sémantique et détails)
    hierarchical_levels: int = 3        # Niveaux hiérarchiques pour la tokenization
    commitment_weight: float = 0.25     # Poids pour la perte d'engagement
    use_context_modulation: bool = True # Utiliser la modulation contextuelle
    modality_alignment_weight: float = 0.5 # Poids pour l'alignement des modalités
    commitment_cost: float = 0.25       # Coût d'engagement pour VQ-VAE
    
    # Dimensions pour chaque modalité
    modality_dims: Dict[str, int] = field(default_factory=lambda: {
            'text': 512,
            'image': 768,
            'audio': 512,
            'video': 1024,
            'graph': 512
        })
    
    # Configurations spécifiques aux encodeurs
    text_encoder_config: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    vision_encoder_config: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)
    audio_encoder_config: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    video_encoder_config: VideoEncoderConfig = field(default_factory=VideoEncoderConfig)
    graph_encoder_config: GraphEncoderConfig = field(default_factory=GraphEncoderConfig)
    
    # Paramètres d'alignement entre modalités
    alignment_dim: int = 768
    num_alignment_heads: int = 8
    alignment_dropout: float = 0.1
    initial_temperature: float = 0.07
    
    # Paramètres pour la hiérarchie du tokenizer
    level_vocab_sizes: List[int] = field(default_factory=lambda: [8192, 4096, 2048])
    level_dims: List[int] = field(default_factory=lambda: [768, 384, 192])
    level_commitment_costs: List[float] = field(default_factory=lambda: [0.25, 0.5, 1.0])
    num_hierarchy_levels: int = 3        # Alias pour hierarchical_levels (rétrocompatibilité)
    
    # Paramètres pour les relations entre tokens
    relation_hidden_dim: int = 512
    num_relation_types: int = 16
    
    # Paramètres du compresseur neural
    input_dim: int = 768
    bottleneck_dim: int = 192
    num_quantizers: int = 4
    codebook_size: int = 8192
    shared_codebook: bool = False
    ema_decay: float = 0.99
    
    # Paramètres d'entraînement
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    total_steps: int = 1000000
    num_epochs: int = 50
    reconstruction_weight: float = 1.0
    contrastive_weight: float = 0.5      # Poids pour la perte contrastive
    max_grad_norm: float = 1.0
    contrastive_temperature: float = 0.07
    
    # Chemin vers un tokenizer pré-entraîné
    pretrained_tokenizer_path: Optional[str] = None


@dataclass
class ModelArchitectureConfig(BaseConfig):
    """Configuration de l'architecture du modèle.
    
    Attributes:
        hidden_size: Dimension des couches cachées
        num_hidden_layers: Nombre de couches cachées
        num_attention_heads: Nombre de têtes d'attention
        intermediate_size: Taille des couches intermédiaires
        hidden_dropout_prob: Taux de dropout des couches cachées
        attention_probs_dropout_prob: Taux de dropout des probabilités d'attention
        max_position_embeddings: Longueur maximale des séquences
        type_vocab_size: Taille du vocabulaire des types de tokens
        initializer_range: Écart type de l'initialisation
        layer_norm_eps: Epsilon pour la normalisation de couche
    """
    # Configuration générale
    hidden_size: int = 256
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    max_seq_length: int = 512
    
    # Configuration de l'encodeur d'entrée
    input_projection_type: str = "minhash_bloom"  # ["minhash_bloom", "ngram_hash"]
    minhash_num_permutations: int = 128
    bloom_filter_size: int = 512
    vocab_size: int = 30000  # Seulement utilisé pour les tokens classiques si nécessaire
    
    # Configuration MLP-Mixer
    num_mixer_layers: int = 6
    token_mixing_hidden_size: int = 512
    channel_mixing_hidden_size: int = 1024
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    activation: str = "gelu"  # ["gelu", "relu", "silu"]
    
    # Configuration mémoire externe
    use_external_memory: bool = True
    memory_size: int = 64
    memory_dim: int = 256
    memory_update_rate: float = 0.1
    memory_levels: int = 3
    
    # Configuration spécifique pour la mémoire hiérarchique
    use_hierarchical_memory: bool = True
    short_term_memory_size: int = 32
    long_term_memory_size: int = 64
    persistent_memory_size: int = 128
    novelty_threshold_stm: float = 0.7
    novelty_threshold_ltm: float = 0.5
    
    # Configuration routage dynamique
    use_dynamic_routing: bool = True
    num_experts: int = 4
    routing_top_k: int = 2
    
    # Configuration de l'adaptateur d'apprentissage continu
    use_continual_adapter: bool = False
    continual_adapter_buffer_size: int = 100
    continual_adapter_rate: float = 0.1
    continual_adapter_drift_threshold: float = 0.5
    continual_adapter_dropout_rate: float = 0.1
    continual_adapter_num_experts: int = 4
    continual_adapter_routing_top_k: int = 2
    continual_adapter_routing_hidden_size: int = 256
    continual_adapter_routing_activation: str = "gelu"
    continual_adapter_routing_layer_norm_epsilon: float = 1e-6
    
    # Configuration pour l'entrée multimodale
    use_multimodal_input: bool = False
    use_cross_modal_attention: bool = True
    cross_modal_num_heads: int = 8
    multimodal_hidden_dim: int = 768
    multimodal_output_dim: int = 0  # Si 0, utilise hidden_size
    multimodal_image_patch_size: int = 16
    multimodal_video_num_sampled_frames: int = 5
    
    # Configuration module symbolique
    use_symbolic_module: bool = False
    symbolic_rules_file: Optional[str] = None
    max_predicate_types: int = 50
    max_entities_in_vocab: int = 200
    
    # Configuration Réseau Bayésien
    use_bayesian_module: bool = False
    num_bayesian_variables: int = 10
    bayesian_network_structure: Optional[List[Tuple[int, int]]] = None
    
    # Configuration quantification
    quantization: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit la configuration en dictionnaire en gérant correctement les champs optionnels.
        
        Returns:
            Dict[str, Any]: Dictionnaire contenant la configuration sérialisée
        """
        output = {}
        
        # Parcourir tous les champs de la classe
        for field_name, field_info in self.__dataclass_fields__.items():
            # Ne pas sérialiser les champs privés
            if field_name.startswith('_'):
                continue
                
            try:
                # Récupérer la valeur du champ
                value = getattr(self, field_name, None)
                
                # Ignorer les valeurs None
                if value is None:
                    continue
                
                # Utiliser la méthode de conversion de la classe parente
                serialized_value = self._convert_to_serializable(value)
                
                # Pour les listes, tuples et dictionnaires vides, on les inclut
                if isinstance(serialized_value, (list, dict, tuple, set)) and not serialized_value:
                    output[field_name] = serialized_value
                    continue
                    
                # Pour les autres types, on vérifie s'ils sont différents de la valeur par défaut
                if hasattr(field_info, 'default'):
                    default_value = field_info.default
                    # Si la valeur est égale à la valeur par défaut, on saute
                    if value == default_value:
                        continue
                
                # Pour les champs sans valeur par défaut ou avec une valeur différente
                output[field_name] = serialized_value
                
            except Exception as e:
                error_type = type(e).__name__
                print(f"Avertissement: Impossible de sérialiser le champ {field_name} (type: {type(value).__name__}): {error_type} - {str(e)}")
                # On inclut quand même le champ avec une valeur d'erreur pour le débogage
                output[f"{field_name}_error"] = f"SerializationError: {error_type} - {str(e)}"
        
        return output


@dataclass
class MemoryConfig:
    """Configuration pour la mémoire du modèle.
    
    Attributes:
        use_external_memory: Activer la mémoire externe
        memory_size: Taille de la mémoire
        memory_dim: Dimension des vecteurs de mémoire
        num_memory_heads: Nombre de têtes pour l'attention sur la mémoire
        memory_update_mechanism: Mécanisme de mise à jour de la mémoire
    """
    use_external_memory: bool = True
    memory_size: int = 64
    memory_dim: int = 256
    num_memory_heads: int = 4
    memory_update_mechanism: str = "gru"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire."""
        return {
            'use_external_memory': self.use_external_memory,
            'memory_size': self.memory_size,
            'memory_dim': self.memory_dim,
            'num_memory_heads': self.num_memory_heads,
            'memory_update_mechanism': self.memory_update_mechanism
        }  # 'gru', 'mlp', 'attention'


@dataclass
class TokenizerConfig:
    """Configuration pour le tokenizer.
    
    Attributes:
        vocab_size: Taille du vocabulaire
        max_length: Longueur maximale des séquences
        padding_side: Côté de padding ('left' ou 'right')
        truncation: Activer la troncature
    """
    vocab_size: int = 50000
    max_length: int = 512
    padding_side: str = "right"
    truncation: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire."""
        return {
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'padding_side': self.padding_side,
            'truncation': self.truncation
        }


@dataclass
class NeuroLiteConfig(BaseConfig):
    """Configuration complète pour un modèle NeuroLite.
    
    Cette classe rassemble toutes les configurations nécessaires pour initialiser
    et entraîner un modèle NeuroLite.
    
    Attributes:
        model_config: Configuration de l'architecture du modèle
        training_config: Configuration pour l'entraînement
        logging_config: Configuration pour la journalisation
        memory_config: Configuration pour la mémoire
        tokenizer_config: Configuration pour le tokenizer
        use_multimodal: Activer le traitement multimodal
        modalities: Liste des modalités supportées
        device: Appareil sur lequel exécuter le modèle
        seed: Graine aléatoire pour la reproductibilité
    """
    # Configurations
    model_config: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    tokenizer_config: TokenizerConfig = field(default_factory=TokenizerConfig)
    
    # Paramètres généraux
    use_multimodal: bool = False
    modalities: List[str] = field(
        default_factory=lambda: ["text", "image", "audio", "video", "graph"]
    )
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit la configuration complète en dictionnaire en gérant correctement les configurations imbriquées.
        
        Returns:
            Dict[str, Any]: Dictionnaire contenant toute la configuration sérialisée
        """
        output = {}
        
        # Liste des configurations à sérialiser avec leur nom d'attribut
        configs_to_serialize = [
            ('model_config', self.model_config),
            ('training_config', self.training_config),
            ('logging_config', self.logging_config),
            ('memory_config', self.memory_config),
            ('tokenizer_config', self.tokenizer_config)
        ]
        
        # Sérialiser chaque configuration
        for key, config in configs_to_serialize:
            try:
                if config is None:
                    output[key] = None
                    continue
                    
                # Essayer d'abord to_dict() si disponible
                if hasattr(config, 'to_dict') and callable(getattr(config, 'to_dict')):
                    try:
                        serialized = config.to_dict()
                        if serialized is not None:  # Ne pas ajouter si la sérialisation a échoué
                            output[key] = serialized
                        continue
                    except Exception as e:
                        print(f"Avertissement: Échec de to_dict() sur {key}: {e}")
                
                # Sinon, essayer de convertir en dict directement
                try:
                    if hasattr(config, '__dict__'):
                        serialized = self._convert_to_serializable(config.__dict__)
                        if serialized:  # Ne pas ajouter si le dictionnaire est vide
                            output[key] = serialized
                        continue
                except Exception as e:
                    print(f"Avertissement: Échec de la conversion de {key}.__dict__: {e}")
                
                # En dernier recours, convertir en chaîne
                output[key] = str(config)
                
            except Exception as e:
                print(f"Erreur critique lors de la sérialisation de {key}: {e}")
                output[key] = None
        
        # Ajouter les champs simples avec des valeurs par défaut sécurisées
        simple_fields = {
            'use_multimodal': getattr(self, 'use_multimodal', False),
            'modalities': getattr(self, 'modalities', []),
            'device': str(getattr(self, 'device', 'cpu')),
            'seed': getattr(self, 'seed', 42)
        }
        
        # Ne pas écraser les valeurs existantes avec des valeurs par défaut
        for key, default_value in simple_fields.items():
            if key not in output:
                output[key] = default_value
        
        # Nettoyer le dictionnaire de sortie
        return {k: v for k, v in output.items() if v is not None}
        
    def __post_init__(self):
        # S'assurer que les configurations sont des instances des bonnes classes
        if not isinstance(self.model_config, ModelArchitectureConfig):
            self.model_config = ModelArchitectureConfig(**self.model_config)
        if not isinstance(self.training_config, TrainingConfig):
            self.training_config = TrainingConfig(**self.training_config)
        if not isinstance(self.logging_config, LoggingConfig):
            self.logging_config = LoggingConfig(**self.logging_config)
        if not isinstance(self.memory_config, MemoryConfig):
            self.memory_config = MemoryConfig(**self.memory_config)
        if not isinstance(self.tokenizer_config, TokenizerConfig):
            self.tokenizer_config = TokenizerConfig(**self.tokenizer_config)
    
    # Méthodes de classe pour les configurations prédéfinies
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "NeuroLiteConfig":
        """Crée une configuration à partir d'un dictionnaire.
        
        Args:
            config_dict: Dictionnaire contenant les paramètres de configuration
            
        Returns:
            Une instance de NeuroLiteConfig
        """
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire.
        
        Returns:
            Un dictionnaire contenant tous les paramètres de configuration
        """
        return {
            "model_config": self.model_config.to_dict() if hasattr(self.model_config, 'to_dict') else vars(self.model_config),
            "training_config": self.training_config.to_dict() if hasattr(self.training_config, 'to_dict') else vars(self.training_config),
            "logging_config": self.logging_config.to_dict() if hasattr(self.logging_config, 'to_dict') else vars(self.logging_config),
            "memory_config": self.memory_config.to_dict() if hasattr(self.memory_config, 'to_dict') else vars(self.memory_config),
            "tokenizer_config": self.tokenizer_config.to_dict() if hasattr(self.tokenizer_config, 'to_dict') else vars(self.tokenizer_config),
            "use_multimodal": self.use_multimodal,
            "modalities": self.modalities,
            "device": self.device,
            "seed": self.seed
        }
    
    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        """Enregistre la configuration dans un répertoire.
        
        Args:
            save_directory: Répertoire de destination
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        output_config_file = os.path.join(save_directory, "config.json")
        with open(output_config_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike]) -> "NeuroLiteConfig":
        """Charge une configuration à partir d'un répertoire ou d'un nom de modèle.
        
        Args:
            pretrained_model_name_or_path: Chemin ou identifiant du modèle
            
        Returns:
            Une instance de NeuroLiteConfig
        """
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        else:
            # Ici, vous pourriez implémenter le téléchargement depuis un dépôt
            raise ValueError(f"Configuration non trouvée dans {pretrained_model_name_or_path}")
        
        with open(config_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        
        config_dict = json.loads(text)
        return cls.from_dict(config_dict)
    
    # Configurations prédéfinies
    
    @classmethod
    def tiny(cls) -> "NeuroLiteConfig":
        """Configuration pour un modèle très léger (~1-2Mo)."""
        model_config = ModelArchitectureConfig(
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=512,
        )
        return cls(model_config=model_config)
    
    @classmethod
    def small(cls) -> "NeuroLiteConfig":
        """Configuration pour un modèle léger (~5-10Mo)."""
        model_config = ModelArchitectureConfig(
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=1024,
        )
        return cls(model_config=model_config)
    
    @classmethod
    def base(cls) -> "NeuroLiteConfig":
        """Configuration pour un modèle de base (~20-30Mo)."""
        return cls()  # Utilise les valeurs par défaut
    
    @classmethod
    def large(cls) -> "NeuroLiteConfig":
        """Configuration pour un grand modèle (~100-200Mo)."""
        model_config = ModelArchitectureConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
        )
        return cls(model_config=model_config)
    
    @classmethod
    def multimodal(cls) -> "NeuroLiteConfig":
        """Configuration pour un modèle multimodal."""
        config = cls.base()
        config.use_multimodal = True
        return config
    
    @classmethod
    def symbolic(cls) -> "NeuroLiteConfig":
        """Configuration pour un modèle avec raisonnement symbolique."""
        config = cls.base()
        # Ajouter des configurations spécifiques au raisonnement symbolique
        return config
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "NeuroLiteConfig":
        """Crée une configuration à partir d'un dictionnaire."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit la configuration complète en dictionnaire en gérant correctement les configurations imbriquées.
        
        Returns:
            Dict[str, Any]: Dictionnaire contenant toute la configuration sérialisée
        """
        output = {}
        for field_name, field_value in self.__dict__.items():
            # Utiliser la méthode to_dict() si disponible, sinon utiliser la conversion générique
            if hasattr(field_value, 'to_dict') and callable(field_value.to_dict):
                output[field_name] = field_value.to_dict()
            else:
                output[field_name] = self._convert_to_serializable(field_value)
        return output
    
    @classmethod
    def tiny(cls) -> "NeuroLiteConfig":
        """Configuration pour un modèle très léger (~1-2Mo)."""
        model_config = ModelArchitectureConfig(
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=512,
        )
        memory_config = MemoryConfig(
            memory_size=32,
            memory_dim=128,
            num_memory_heads=2
        )
        return cls(model_config=model_config, memory_config=memory_config)
    
    @classmethod
    def small(cls) -> "NeuroLiteConfig":
        """Configuration pour un modèle léger (~5-10Mo)."""
        model_config = ModelArchitectureConfig(
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=1024,
        )
        memory_config = MemoryConfig(
            memory_size=64,
            memory_dim=256,
            num_memory_heads=4
        )
        return cls(model_config=model_config, memory_config=memory_config)
    
    @classmethod
    def base(cls) -> "NeuroLiteConfig":
        """Configuration pour un modèle de base (~20-30Mo)."""
        return cls()  # Utilise les valeurs par défaut
    
    @classmethod
    def large(cls) -> "NeuroLiteConfig":
        """Configuration pour un grand modèle (~100-200Mo)."""
        model_config = ModelArchitectureConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
        )
        memory_config = MemoryConfig(
            memory_size=256,
            memory_dim=1024,
            num_memory_heads=8
        )
        return cls(model_config=model_config, memory_config=memory_config)
    
    @classmethod
    def base_symbolic(cls) -> "NeuroLiteConfig":
        """Configuration pour un modèle de base avec raisonnement symbolique."""
        config = cls.base()
        # Ajouter des configurations spécifiques au raisonnement symbolique
        return config
