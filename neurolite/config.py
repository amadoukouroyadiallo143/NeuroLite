"""
Configuration pour les modèles NeuroLite
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Union, Any


@dataclass
class NeuroLiteConfig:
    """Configuration pour l'architecture NeuroLite"""
    
    # Dimension du modèle
    hidden_size: int = 256
    
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
    memory_size: int = 64  # Nombre de slots de mémoire
    memory_dim: int = 256  # Dimension de chaque slot
    memory_update_rate: float = 0.1  # Taux de mise à jour de la mémoire
    
    # Configuration module symbolique
    use_symbolic_module: bool = False
    symbolic_rules_file: Optional[str] = None
    max_predicate_types: int = 50 # For NeuralSymbolicLayer's predicate vocab
    max_entities_in_vocab: int = 200 # For NeuralSymbolicLayer's entity vocab
    
    # Configuration Réseau Bayésien
    use_bayesian_module: bool = False # To control adding BayesianBeliefNetwork in NeuroLiteModel
    num_bayesian_variables: int = 10
    bayesian_network_structure: Optional[List[Tuple[int, int]]] = None # List of (parent_idx, child_idx)

    # Configuration routage dynamique
    use_dynamic_routing: bool = True
    num_experts: int = 4
    routing_top_k: int = 2  # Nombre d'experts à activer par token
    
    # Configuration de séquence
    max_seq_length: int = 512
    
    # Configuration quantification
    quantization: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "NeuroLiteConfig":
        """Crée une configuration à partir d'un dictionnaire"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire"""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def tiny(cls) -> "NeuroLiteConfig":
        """Configuration pour un modèle très léger (~1-2Mo)"""
        return cls(
            hidden_size=128,
            num_mixer_layers=4,
            token_mixing_hidden_size=256,
            channel_mixing_hidden_size=512,
            memory_size=32,
            memory_dim=128,
            num_experts=2,
        )
    
    @classmethod
    def small(cls) -> "NeuroLiteConfig":
        """Configuration pour un modèle léger (~5-10Mo)"""
        return cls(
            hidden_size=256,
            num_mixer_layers=6,
            token_mixing_hidden_size=512, 
            channel_mixing_hidden_size=1024,
            memory_size=64,
            memory_dim=256,
            num_experts=4,
        )
    
    @classmethod
    def base(cls) -> "NeuroLiteConfig":
        """Configuration pour un modèle de base (~20-30Mo)"""
        return cls(
            hidden_size=384,
            num_mixer_layers=8,
            token_mixing_hidden_size=768,
            channel_mixing_hidden_size=1536,
            memory_size=128,
            memory_dim=384,
            num_experts=6,
        )

    @classmethod
    def base_symbolic(cls) -> "NeuroLiteConfig":
        """Configuration pour un modèle de base avec module symbolique activé."""
        config = cls.base()
        config.use_symbolic_module = True
        config.symbolic_rules_file = "rules.json"  # Default rules file
        # Add Bayesian module related defaults if base_symbolic should also use it
        config.use_bayesian_module = True 
        config.num_bayesian_variables = 10
        # Example: A -> B, B -> C  (0->1, 1->2)
        config.bayesian_network_structure = [(0,1), (1,2)] 
        return config
