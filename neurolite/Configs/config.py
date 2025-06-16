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
        """
        Crée une instance de configuration (dataclass) à partir d'un dictionnaire,
        en gérant récursivement les dataclasses imbriquées.
        """
        # Créer une copie pour éviter de modifier le dictionnaire original
        config_dict = deepcopy(config_dict)

        # Itérer sur les champs de la dataclass cible
        for field_info in fields(cls):
            field_name = field_info.name
            field_type = field_info.type
            
            # Gérer les types 'Optional' (Union[T, None])
            is_optional = hasattr(field_type, '__origin__') and field_type.__origin__ is Union
            if is_optional:
                # Extraire le type réel, ex: de Optional[MMImageEncoderConfig] on veut MMImageEncoderConfig
                actual_type = next((t for t in field_type.__args__ if t is not type(None)), None)
            else:
                actual_type = field_type

            # Si le champ est une dataclass et que nous avons un dictionnaire pour lui...
            if hasattr(actual_type, '__dataclass_fields__') and field_name in config_dict and isinstance(config_dict[field_name], dict):
                # ...appeler récursivement from_dict pour créer l'instance imbriquée
                config_dict[field_name] = actual_type.from_dict(config_dict[field_name])

        # Filtrer les clés du dictionnaire pour ne garder que celles qui sont des champs de la dataclass
        valid_keys = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}

        return cls(**filtered_dict)
    
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
    output_dir: str = "./outputs"
    logging_dir: str = "logs"
    
    # Paramètres d'entraînement
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    warmup_steps: int = 0
    weight_decay: float = 0.01
    learning_rate: float = 5e-5
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Précision mixte
    fp16: bool = True
    fp16_opt_level: str = "O1"
    
    # Fréquences
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    save_total_limit: int = 3
    
    # Chargement du meilleur modèle
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Métriques à calculer
    metrics: List[str] = field(
        default_factory=lambda: ["accuracy", "f1", "precision", "recall"]
    )
    
    lr_scheduler_type: str = "linear"
    num_warmup_steps: int = 0
    evaluation_strategy: str = "steps"  # "steps" ou "epoch"


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
class MMEncoderConfigBase(BaseConfig):
    """Configuration de base pour les encodeurs de MultimodalProjection."""
    num_layers: int = 4
    hidden_dim: int = 256 # Dimension interne de l'encodeur spécifique à la modalité
    output_dim: int = 512 # Dimension de sortie projetée (peut être différent de hidden_dim du modèle global)
    num_attention_heads: int = 4
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-6
    use_ssm: bool = False
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_expand_factor: int = 2
    ssm_dt_rank: Union[int, str] = 'auto'
    ssm_bias: bool = False
    ssm_conv_bias: bool = True
    ssm_bidirectional: bool = False

@dataclass
class MMTextEncoderConfig(MMEncoderConfigBase):
    """Configuration spécifique pour l'encodeur de texte multimodal."""
    vocab_size: int = 32000
    max_position_embeddings: int = 2048

@dataclass
class MMImageEncoderConfig(MMEncoderConfigBase):
    """Configuration spécifique pour l'encodeur d'image multimodal."""
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    use_cls_token: bool = True

@dataclass
class MMAudioEncoderConfig(MMEncoderConfigBase):
    """Configuration spécifique pour l'encodeur audio multimodal."""
    sampling_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    max_audio_length_ms: int = 30000

@dataclass
class MMVideoEncoderConfig(MMEncoderConfigBase):
    """Configuration spécifique pour l'encodeur vidéo multimodal."""
    image_encoder_config: MMImageEncoderConfig = field(default_factory=MMImageEncoderConfig)
    num_frames_input: int = 16
    temporal_patch_size: int = 2
    temporal_num_layers: int = 2
    temporal_num_attention_heads: int = 4
    temporal_mlp_ratio: float = 4.0
    temporal_use_ssm: bool = False

@dataclass
class MMGraphEncoderConfig(MMEncoderConfigBase):
    """Configuration spécifique pour l'encodeur de graphe multimodal."""
    node_feature_dim: int = 128
    edge_feature_dim: Optional[int] = 64
    num_node_types: int = 10
    num_edge_types: int = 5
    gnn_type: str = "GAT"
    pooling_method: str = "mean" # Options: "mean", "attention"

@dataclass
class MMDecoderConfigBase(BaseConfig):
    """Configuration de base pour les décodeurs de MultimodalGeneration."""
    num_layers: int = 4
    hidden_dim: int = 256
    input_dim: int = 512
    num_attention_heads: int = 4
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-6
    use_ssm: bool = False
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_expand_factor: int = 2
    ssm_dt_rank: Union[int, str] = 'auto'
    ssm_bias: bool = False
    ssm_conv_bias: bool = True
    ssm_bidirectional: bool = False

@dataclass
class MMTextDecoderConfig(MMDecoderConfigBase):
    """Configuration spécifique pour le décodeur de texte multimodal."""
    name: str = "text_decoder"
    input_dim: int = 512
    embedding_dim: int = 256
    hidden_dim: int = 1024 # dim_feedforward
    vocab_size: int = 32000
    max_seq_len: int = 512
    num_heads: int = 4 # n_head
    num_layers: int = 4 # n_layers
    dropout_rate: float = 0.1 # dropout
    use_positional_encoding: bool = True
    use_ssm_layers: bool = False
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_expand_factor: int = 2
    output_activation: str = "softmax"
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

@dataclass
class MMImageDecoderConfig(MMDecoderConfigBase):
    """Configuration spécifique pour le décodeur d'image multimodal."""
    name: str = "image_decoder"
    input_dim: int = 256
    embedding_dim: int = 256
    hidden_dim: int = 1024
    num_heads: int = 4
    num_layers: int = 4
    dropout_rate: float = 0.1
    use_positional_encoding: bool = True
    use_ssm_layers: bool = False
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_expand_factor: int = 2
    patch_size: int = 16
    initial_size: int = 7
    initial_channels: int = 256
    output_channels: int = 3
    num_upsamples: int = 5
    use_residual: bool = True
    final_activation: str = "sigmoid"
    target_image_size: int = 224

@dataclass
class MMAudioDecoderConfig(MMDecoderConfigBase):
    """Configuration spécifique pour le décodeur audio multimodal."""
    name: str = "audio_decoder"
    input_dim: int = 256
    embedding_dim: int = 256
    hidden_dim: int = 1024
    num_heads: int = 4
    num_layers: int = 4
    dropout_rate: float = 0.1
    use_positional_encoding: bool = True
    use_ssm_layers: bool = False
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_expand_factor: int = 2
    initial_length: int = 16
    initial_channels: int = 256
    output_channels: int = 1
    num_upsamples: int = 5
    use_residual: bool = True
    n_mels: int = 80
    upsample_strides: tuple = (4, 4)
    final_activation: str = "tanh"
    target_sampling_rate: int = 16000
    output_length: int = 32000 # Ajout pour la génération

@dataclass
class MMVideoDecoderConfig(MMDecoderConfigBase):
    """Configuration spécifique pour le décodeur vidéo multimodal."""
    name: str = "video_decoder"
    input_dim: int = 256
    embedding_dim: int = 256
    hidden_dim: int = 1024
    num_heads: int = 4
    num_temporal_layers: int = 2
    num_spatial_layers: int = 4
    dropout_rate: float = 0.1
    use_positional_encoding: bool = True
    use_ssm_layers: bool = False
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_expand_factor: int = 2
    max_frames: int = 32
    image_decoder_config: MMImageDecoderConfig = field(default_factory=MMImageDecoderConfig)
    
    # Dimensions de la représentation latente projetée
    initial_channels: int = 256
    initial_time: int = 4
    initial_size: int = 4

    # Paramètres des blocs de suréchantillonnage
    num_temporal_upsamples: int = 2
    num_spatial_upsamples: int = 4
    use_residual: bool = True

    # Paramètres de sortie
    output_channels: int = 3
    output_time: int = 16
    output_size: int = 224
    final_activation: str = "tanh" # "tanh", "sigmoid", ou "identity"


@dataclass
class MMGraphDecoderConfig(MMDecoderConfigBase):
    """Configuration spécifique pour le décodeur de graphe multimodal."""
    name: str = "graph_decoder"
    input_dim: int = 256
    embedding_dim: int = 256
    hidden_dim: int = 1024
    num_heads: int = 4
    num_layers: int = 4
    dropout_rate: float = 0.1
    use_positional_encoding: bool = True
    use_ssm_layers: bool = False
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_expand_factor: int = 2
    max_nodes: int = 100
    node_feature_dim: int = 128
    use_residual: bool = True # Ajout pour les couches de décodeur
    output_activation_nodes: str = "linear"
    target_edge_feature_dim: Optional[int] = 64
    max_edges_output: int = 100
    generation_strategy: str = "autoregressive_node_edge"


@dataclass
class TokenizerConfig(BaseConfig):
    """Configuration complète pour le NeuroLiteTokenizer."""
    # --- Configuration générale ---
    hidden_size: int = 512
    tokenizer_type: str = 'bpe'

    # --- Configuration du Tokenizer de texte (BPE) ---
    vocab_size: int = 50000

    # --- Configuration des modules multimodaux ---
    projection_dropout: float = 0.1
    
    # --- Configuration du compresseur (NeuralCompressor & ResidualVQ) ---
    compressor_bottleneck_dim: int = 256
    num_quantizers: int = 4
    codebook_size: int = 1024
    commitment_weight: float = 0.25
    ema_decay: float = 0.99

    # --- Poids pour la fonction de perte du tokenizer ---
    reconstruction_weight: float = 1.0
    alignment_weight: float = 0.1
    contrastive_temperature: float = 0.1


@dataclass
class ModelArchitectureConfig(BaseConfig):
    """Configuration de l'architecture du modèle."""
    hidden_size: int = 256
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 1024
    max_seq_length: int = 1024
    
    # --- Flags pour les modules principaux ---
    use_metacontroller: bool = False
    use_causal_reasoning: bool = False
    use_multimodal_input: bool = False
    use_external_memory: bool = False
    use_hierarchical_memory: bool = False
    use_dynamic_routing: bool = False
    use_ssm_layers: bool = False
    use_fnet_layers: bool = False

    # --- Paramètres des modules ---
    # SSM
    ssm_layer_indices: Optional[List[int]] = None
    ssm_layer_frequency: Optional[int] = None
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_expand_factor: int = 2
    # FNet
    fnet_layer_indices: Optional[List[int]] = None
    fnet_layer_frequency: Optional[int] = None
    # Dynamic Routing (MoE)
    num_experts: int = 4
    routing_top_k: int = 2
    # Memory
    memory_size: int = 64
    memory_dim: int = 256
    short_term_memory_size: int = 32
    long_term_memory_size: int = 64
    persistent_memory_size: int = 128
    # Multimodal
    multimodal_output_dim: Optional[int] = None
    multimodal_hidden_dim: int = 256
    use_cross_modal_attention: bool = True
    cross_modal_num_heads: int = 8
    attention_probs_dropout_prob: float = 0.1
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    activation: str = "gelu"
    token_mixing_hidden_size: int = 512
    channel_mixing_hidden_size: int = 1024

    # --- Flags et paramètres pour les modules de raisonnement ---
    use_symbolic_module: bool = False
    use_advanced_reasoning: bool = False
    use_bayesian_module: bool = False
    use_planning_module: bool = False
    symbolic_dim: int = 64
    num_inference_steps: int = 3
    symbolic_rules_file: Optional[str] = None
    num_bayesian_variables: int = 0 # Mettre à 0 pour désactiver
    num_planning_steps: int = 5
    plan_dim: int = 64

    # --- Flags et paramètres pour l'apprentissage continu ---
    use_continual_adapter: bool = False
    adapter_bottleneck_dim: int = 64  # Nouvelle valeur, dimension du bottleneck dans l'adaptateur
    task_embedding_dim: int = 16    # Nouvelle valeur, dimension de l'embedding de tâche

    # Configurations granulaires pour les encodeurs multimodaux
    mm_text_encoder_config: Optional[MMTextEncoderConfig] = None
    mm_image_encoder_config: Optional[MMImageEncoderConfig] = None
    mm_audio_encoder_config: Optional[MMAudioEncoderConfig] = None
    mm_video_encoder_config: Optional[MMVideoEncoderConfig] = None
    mm_graph_encoder_config: Optional[MMGraphEncoderConfig] = None

    # Configurations granulaires pour les décodeurs multimodaux (si génération multimodale activée)
    mm_text_decoder_config: Optional[MMTextDecoderConfig] = None
    mm_image_decoder_config: Optional[MMImageDecoderConfig] = None
    mm_audio_decoder_config: Optional[MMAudioDecoderConfig] = None
    mm_video_decoder_config: Optional[MMVideoDecoderConfig] = None
    mm_graph_decoder_config: Optional[MMGraphDecoderConfig] = None

    def __post_init__(self):
        """Re-instancie récursivement les dataclasses à partir des dictionnaires."""
        for field_name, field_type in self.__annotations__.items():
            current_value = getattr(self, field_name)
            
            # Le type est souvent une Union (ex: Optional[MMTextConfig]), il faut trouver le bon type
            actual_type = None
            if hasattr(field_type, '__args__'):
                # Trouver le premier type non-None dans Union[... , NoneType]
                possible_types = [t for t in field_type.__args__ if t is not type(None)]
                if possible_types:
                    actual_type = possible_types[0]
            else:
                actual_type = field_type

            if isinstance(current_value, dict) and actual_type and hasattr(actual_type, '__dataclass_fields__'):
                new_value = actual_type(**current_value)
                setattr(self, field_name, new_value)


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
class MemoryConfig(BaseConfig):
    """Configuration pour la mémoire du modèle."""
    use_external_memory: bool = True
    memory_size: int = 64
    memory_dim: int = 256
    num_memory_heads: int = 4
    dropout_rate: float = 0.1
    hidden_size: int = 256 # Doit correspondre à model_config.hidden_size
    persistent_memory_size: int = 128
    
    # Paramètres de récupération pour la recherche
    k_top_stm: int = 5
    k_top_pm: int = 5
    memory_chunk_size: int = 512
    retrieval_strategy: str = "cosine_similarity" # 'attention_retrieval' ou 'cosine_similarity'
    similarity_exponent: float = 1.0


@dataclass
class LongTermMemoryConfig(BaseConfig):
    """Configuration pour la mémoire à long terme.
    
    Attributes:
        enabled: Activer la mémoire à long terme
        memory_size: Taille de la mémoire à long terme (nombre d'entrées)
        dimension: Dimension des vecteurs de mémoire
        update_rate: Taux de mise à jour de la mémoire
        similarity_threshold: Seuil de similarité pour la fusion d'entrées
        persistence_path: Chemin pour la sauvegarde/chargement de la mémoire persistante
        retrieval_strategy: Stratégie de récupération ('cosine', 'dot_product', 'attention')
        top_k: Nombre maximum d'entrées récupérées
        pruning_strategy: Stratégie d'élagage ('lru', 'least_similar', 'random')
    """
    enabled: bool = False
    memory_size: int = 1024
    dimension: int = 256
    update_rate: float = 0.05
    similarity_threshold: float = 0.7
    persistence_path: Optional[str] = None
    retrieval_strategy: str = "cosine"
    top_k: int = 5
    pruning_strategy: str = "lru"


@dataclass
class ReasoningConfig(BaseConfig):
    """Configuration pour les modules de raisonnement."""
    use_symbolic_module: bool = False
    use_causal_reasoning: bool = False
    use_planning_module: bool = False
    use_bayesian_module: bool = False

    # Paramètres pour NeurosymbolicReasoner
    symbolic_dim: int = 64
    num_inference_steps: int = 3
    max_facts: int = 100

    # Paramètres pour CausalInferenceEngine (si nécessaire)
    causal_graph_path: Optional[str] = None

    # Paramètres pour StructuredPlanner
    num_planning_steps: int = 5
    plan_dim: int = 64


@dataclass
class NeuroLiteConfig(BaseConfig):
    """Configuration complète pour un modèle NeuroLite."""
    model_config: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    long_term_memory_config: LongTermMemoryConfig = field(default_factory=LongTermMemoryConfig)
    tokenizer_config: TokenizerConfig = field(default_factory=TokenizerConfig)
    reasoning_config: ReasoningConfig = field(default_factory=ReasoningConfig)
    
    use_multimodal: bool = False
    modalities: List[str] = field(default_factory=lambda: ["text", "image", "audio", "video", "graph"])
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    use_torch_compile: bool = field(default=False, metadata={"help": "Activer torch.compile pour optimiser le modèle."})
    torch_compile_mode: Optional[str] = field(default="default", metadata={"help": "Mode pour torch.compile."})
    torch_compile_backend: Optional[str] = field(default=None, metadata={"help": "Backend pour torch.compile."})
    torch_compile_options: Optional[Dict[str, Any]] = field(default=None, metadata={"help": "Options pour torch.compile."})
    
    def __post_init__(self):
        """Re-instancie récursivement les dataclasses à partir des dictionnaires après le chargement."""
        for field_name, field_type in self.__annotations__.items():
            # Obtenir la valeur actuelle du champ
            current_value = getattr(self, field_name)
            
            # Vérifier si la valeur est un dictionnaire et si le type attendu est une dataclass
            # (On vérifie s'il a __dataclass_fields__ pour être sûr)
            if isinstance(current_value, dict) and hasattr(field_type, '__dataclass_fields__'):
                # Créer une instance de la dataclass à partir du dictionnaire
                # Gérer le cas où le champ est une Union/Optional
                actual_type = None
                if hasattr(field_type, '__args__'):
                    possible_types = [t for t in field_type.__args__ if t is not type(None)]
                    if possible_types:
                        actual_type = possible_types[0]
                else:
                    actual_type = field_type

                if actual_type and hasattr(actual_type, '__dataclass_fields__'):
                    try:
                        new_value = actual_type(**current_value)
                        setattr(self, field_name, new_value)
                        if hasattr(new_value, '__post_init__'):
                            new_value.__post_init__()
                    except TypeError as e:
                        # Gère les cas où current_value a des clés qui ne correspondent pas aux champs de actual_type
                        print(f"Avertissement lors de la ré-instanciation de {actual_type.__name__}: {e}. Utilisation des valeurs par défaut.")
                        setattr(self, field_name, actual_type())
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "NeuroLiteConfig":
        config_dict = deepcopy(config_dict)
        # Instantiate nested dataclasses from their dict representations
        model_config = ModelArchitectureConfig.from_dict(config_dict.pop('model_config', {}))
        training_config = TrainingConfig.from_dict(config_dict.pop('training_config', {}))
        logging_config = LoggingConfig.from_dict(config_dict.pop('logging_config', {}))
        memory_config = MemoryConfig.from_dict(config_dict.pop('memory_config', {}))
        long_term_memory_config = LongTermMemoryConfig.from_dict(config_dict.pop('long_term_memory_config', {}))
        tokenizer_config = TokenizerConfig.from_dict(config_dict.pop('tokenizer_config', {}))
        reasoning_config = ReasoningConfig.from_dict(config_dict.pop('reasoning_config', {}))
        
        return cls(
            model_config=model_config,
            training_config=training_config,
            logging_config=logging_config,
            memory_config=memory_config,
            long_term_memory_config=long_term_memory_config,
            tokenizer_config=tokenizer_config,
            reasoning_config=reasoning_config,
            **config_dict
        )
    
    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        output_config_file = os.path.join(save_directory, "config.json")
        with open(output_config_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike]) -> "NeuroLiteConfig":
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        else:
            raise ValueError(f"Configuration non trouvée dans {pretrained_model_name_or_path}")
        
        with open(config_file, "r", encoding="utf-8") as reader:
            config_dict = json.load(reader)
        
        return cls.from_dict(config_dict)
    
    # --- Predefined static configurations ---
    @classmethod
    def tiny(cls) -> "NeuroLiteConfig":
        """Configuration pour un modèle très léger (~1-2Mo)."""
        model_config = ModelArchitectureConfig(hidden_size=128, num_hidden_layers=4, num_attention_heads=4, intermediate_size=512)
        return cls(model_config=model_config)
    
    @classmethod
    def small(cls) -> "NeuroLiteConfig":
        """Configuration pour un modèle léger (~5-10Mo)."""
        model_config = ModelArchitectureConfig(hidden_size=256, num_hidden_layers=6, num_attention_heads=8, intermediate_size=1024)
        return cls(model_config=model_config)
    
    @classmethod
    def base(cls) -> "NeuroLiteConfig":
        """Configuration pour un modèle de base (~20-30Mo)."""
        return cls()
    
    @classmethod
    def large(cls) -> "NeuroLiteConfig":
        """Configuration pour un grand modèle (~100-200Mo)."""
        model_config = ModelArchitectureConfig(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096)
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
        # Specific symbolic reasoning configs would be set here
        return config
