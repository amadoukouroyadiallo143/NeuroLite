"""
Configuration pour le tokenizer multimodal NeuroLite.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any


@dataclass
class EncoderConfig:
    """Configuration de base pour les encodeurs de modalités."""
    hidden_size: int = 256
    num_layers: int = 3
    dropout_rate: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-6


@dataclass
class TextEncoderConfig(EncoderConfig):
    """Configuration pour l'encodeur de texte."""
    vocab_size: int = 50000
    max_position_embeddings: int = 2048
    embedding_size: int = 256
    use_learned_position_embeddings: bool = True
    use_relative_position_bias: bool = True
    share_embeddings_with_lm_head: bool = True


@dataclass
class ImageEncoderConfig(EncoderConfig):
    """Configuration pour l'encodeur d'image."""
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    use_adaptive_patches: bool = True
    use_patch_dropout: bool = True
    patch_dropout_rate: float = 0.1
    use_cls_token: bool = True


@dataclass
class AudioEncoderConfig(EncoderConfig):
    """Configuration pour l'encodeur audio."""
    sampling_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    max_audio_length_ms: int = 30000
    use_spectrogram_augmentation: bool = True


@dataclass
class VideoEncoderConfig(EncoderConfig):
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
class GraphEncoderConfig(EncoderConfig):
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
class QuantizerConfig:
    """Configuration pour les quantificateurs vectoriels."""
    n_embeddings: int = 8192
    embedding_dim: int = 256
    commitment_cost: float = 0.25
    use_ema_updates: bool = True
    ema_decay: float = 0.99
    restart_unused_codes: bool = True
    threshold_ema_dead_code: float = 1e-5


@dataclass
class TokenizerConfig:
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
    commitment_weight: float = 0.25      # Poids pour la perte d'engagement
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
