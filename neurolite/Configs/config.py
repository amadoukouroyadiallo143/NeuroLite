"""
NeuroLite AGI v2.0 - Configuration Avancée Enterprise
Architecture de configuration industrielle avec optimisations de performance.
"""

import os
import json
import torch
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from pathlib import Path

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """Niveaux d'optimisation du modèle."""
    DEVELOPMENT = "dev"      # Maximum de débuggage, pas d'optimisations
    PRODUCTION = "prod"      # Optimisations maximales
    INFERENCE = "inference"  # Optimisé pour l'inférence uniquement
    EDGE = "edge"           # Optimisé pour déploiement edge/mobile

class PrecisionMode(Enum):
    """Modes de précision numérique."""
    FP32 = "fp32"           # Précision maximale
    FP16 = "fp16"           # Mixed precision
    BF16 = "bf16"           # BFloat16 pour TPU
    INT8 = "int8"           # Quantification INT8

class DeviceStrategy(Enum):
    """Stratégies de déploiement multi-device."""
    SINGLE_GPU = "single_gpu"
    MULTI_GPU = "multi_gpu" 
    DISTRIBUTED = "distributed"
    TPU = "tpu"
    CPU_ONLY = "cpu_only"
    HYBRID = "hybrid"

@dataclass
class PerformanceConfig:
    """Configuration de performance avancée."""
    # Optimisations mémoire
    gradient_checkpointing: bool = True
    memory_efficient_attention: bool = True
    activation_checkpointing: bool = True
    cpu_offload: bool = False
    
    # Optimisations compute
    compile_model: bool = True
    use_flash_attention: bool = True
    fused_ops: bool = True
    tensorrt_optimization: bool = False
    
    # Parallélisation
    data_parallel: bool = True
    tensor_parallel: bool = False
    pipeline_parallel: bool = False
    sequence_parallel: bool = False
    
    # Caching et précomputation
    kv_cache_enabled: bool = True
    static_cache_size: Optional[int] = None
    precompute_freqs_cis: bool = True
    
    # Precision et quantification
    precision_mode: PrecisionMode = PrecisionMode.FP16
    dynamic_loss_scaling: bool = True
    quantization_enabled: bool = False
    
    # Profiling et monitoring
    enable_profiling: bool = False
    memory_profiling: bool = False
    compute_profiling: bool = False

@dataclass
class ModelConfig:
    """Configuration du modèle NeuroLite optimisée."""
    # Architecture de base
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    
    # Paramètres SSM optimisés
    ssm_state_size: int = 16
    ssm_conv_kernel: int = 4
    ssm_expand_factor: int = 2
    ssm_dt_rank: Union[int, str] = "auto"
    
    # Paramètres de conscience
    consciousness_levels: int = 5
    consciousness_hidden_size: int = 256
    introspection_depth: int = 3
    
    # Mémoire infinie
    memory_capacity_mb: int = 1024
    working_memory_size: int = 32
    episodic_memory_retention_hours: int = 168  # 1 semaine
    
    # World model
    world_simulation_steps: int = 10
    physics_simulation_enabled: bool = True
    environment_complexity: str = "medium"  # low, medium, high
    
    # Optimisations réseau
    layer_norm_epsilon: float = 1e-6
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    residual_dropout: float = 0.1
    
    # Initialisation
    initializer_range: float = 0.02
    layer_norm_init: float = 1.0
    embed_init_std: float = 0.02
    
    # Contraintes de performance
    max_sequence_length: int = 4096
    max_batch_size: int = 32
    max_memory_gb: float = 24.0

@dataclass
class ConsciousnessConfig:
    """Configuration du système de conscience."""
    enabled: bool = True
    max_level: str = "metacognitive"
    attention_mechanism: str = "multi_head"  # multi_head, flash, memory_efficient
    self_model_enabled: bool = True
    introspection_enabled: bool = True
    metacognition_enabled: bool = True
    
    # Paramètres d'introspection
    introspection_frequency: float = 0.1  # 10% des forward passes
    self_reflection_depth: int = 3
    coherence_threshold: float = 0.7
    
    # Méta-cognition
    meta_learning_rate: float = 1e-5
    meta_adaptation_enabled: bool = True

@dataclass
class MemoryConfig:
    """Configuration du système de mémoire."""
    # Types de mémoire activés
    enable_episodic: bool = True
    enable_semantic: bool = True
    enable_procedural: bool = True
    enable_working: bool = True
    
    # Capacités par type
    working_memory_capacity: int = 7  # 7±2 règle de Miller
    episodic_memory_mb: int = 512
    semantic_memory_mb: int = 1024
    procedural_memory_mb: int = 256
    
    # Paramètres de récupération
    similarity_threshold: float = 0.8
    max_retrieval_results: int = 10
    time_decay_enabled: bool = True
    consolidation_enabled: bool = True
    
    # Compression et archivage
    auto_compression: bool = True
    compression_threshold_age_hours: int = 72
    archival_enabled: bool = True

@dataclass
class ReasoningConfig:
    """Configuration du moteur de raisonnement."""
    enabled_types: List[str] = field(default_factory=lambda: [
        "deductive", "inductive", "abductive", "analogical", "causal"
    ])
    
    # Paramètres de raisonnement
    max_reasoning_steps: int = 10
    confidence_threshold: float = 0.6
    enable_symbolic: bool = True
    enable_neural: bool = True
    
    # Optimisations
    parallel_reasoning: bool = True
    reasoning_cache_enabled: bool = True
    incremental_reasoning: bool = True

@dataclass
class PlanningConfig:
    """Configuration du système de planification."""
    enable_strategic: bool = True
    enable_tactical: bool = True
    enable_operational: bool = True
    
    # Horizons de planification
    short_term_horizon: int = 10
    medium_term_horizon: int = 50
    long_term_horizon: int = 200
    
    # World model
    physics_accuracy: str = "medium"  # low, medium, high
    environment_modeling: bool = True
    uncertainty_modeling: bool = True

@dataclass
class DeviceConfig:
    """Configuration spécifique au dispositif."""
    strategy: DeviceStrategy = DeviceStrategy.SINGLE_GPU
    device_ids: Optional[List[int]] = None
    master_port: int = 29500
    
    # Gestion mémoire
    memory_fraction: float = 0.9
    reserved_memory_gb: float = 2.0
    swap_enabled: bool = True
    
    # Optimisations spécifiques
    cuda_benchmark: bool = True
    cudnn_deterministic: bool = False
    allow_tf32: bool = True

@dataclass
class NeuroLiteConfig:
    """Configuration maître NeuroLite AGI v2.0."""
    
    # Configurations principales
    model_config: ModelConfig = field(default_factory=ModelConfig)
    performance_config: PerformanceConfig = field(default_factory=PerformanceConfig)
    device_config: DeviceConfig = field(default_factory=DeviceConfig)
    
    # Modules spécialisés
    consciousness_config: ConsciousnessConfig = field(default_factory=ConsciousnessConfig)
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    reasoning_config: ReasoningConfig = field(default_factory=ReasoningConfig)
    planning_config: PlanningConfig = field(default_factory=PlanningConfig)
    
    # Métadonnées
    optimization_level: OptimizationLevel = OptimizationLevel.DEVELOPMENT
    version: str = "2.0.0"
    environment: str = "development"
    
    def __post_init__(self):
        """Post-traitement et validation de la configuration."""
        self._validate_config()
        self._apply_optimization_level()
        self._setup_device_strategy()
        self._optimize_for_hardware()
        
        logger.info(f"NeuroLite Config v{self.version} initialisée - {self.optimization_level.value}")
    
    def _validate_config(self):
        """Valide la cohérence de la configuration."""
        # Validation des contraintes mémoire
        total_memory_mb = (
            self.memory_config.episodic_memory_mb +
            self.memory_config.semantic_memory_mb +
            self.memory_config.procedural_memory_mb
        )
        
        if total_memory_mb > self.model_config.memory_capacity_mb:
            logger.warning(f"Mémoire totale ({total_memory_mb}MB) > capacité ({self.model_config.memory_capacity_mb}MB)")
        
        # Validation des dimensions d'attention
        if self.model_config.hidden_size % self.model_config.num_attention_heads != 0:
            raise ValueError("hidden_size doit être divisible par num_attention_heads")
        
        # Validation précision/device
        if self.performance_config.precision_mode == PrecisionMode.BF16 and self.device_config.strategy != DeviceStrategy.TPU:
            logger.warning("BF16 recommandé uniquement pour TPU")
    
    def _apply_optimization_level(self):
        """Applique les optimisations selon le niveau."""
        if self.optimization_level == OptimizationLevel.PRODUCTION:
            # Optimisations maximales pour production
            self.performance_config.compile_model = True
            self.performance_config.use_flash_attention = True
            self.performance_config.fused_ops = True
            self.performance_config.gradient_checkpointing = True
            self.performance_config.precision_mode = PrecisionMode.FP16
            
            # Réduction des capacités pour performance
            self.model_config.num_layers = min(self.model_config.num_layers, 24)
            self.consciousness_config.introspection_frequency = 0.05
            
        elif self.optimization_level == OptimizationLevel.INFERENCE:
            # Optimisé pour inférence uniquement
            self.performance_config.gradient_checkpointing = False
            self.performance_config.kv_cache_enabled = True
            self.performance_config.quantization_enabled = True
            
        elif self.optimization_level == OptimizationLevel.EDGE:
            # Optimisé pour edge computing
            self.model_config.hidden_size = min(self.model_config.hidden_size, 512)
            self.model_config.num_layers = min(self.model_config.num_layers, 8)
            self.performance_config.precision_mode = PrecisionMode.INT8
            self.performance_config.cpu_offload = True
    
    def _setup_device_strategy(self):
        """Configure la stratégie multi-device."""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            if gpu_count > 1 and self.device_config.strategy == DeviceStrategy.SINGLE_GPU:
                self.device_config.strategy = DeviceStrategy.MULTI_GPU
                logger.info(f"Auto-détection: {gpu_count} GPUs disponibles")
            
            if gpu_memory < 8.0:  # Moins de 8GB
                self.performance_config.cpu_offload = True
                logger.warning("Mémoire GPU limitée - activation CPU offload")
        else:
            self.device_config.strategy = DeviceStrategy.CPU_ONLY
            logger.info("GPU non disponible - basculement CPU")
    
    def _optimize_for_hardware(self):
        """Optimise automatiquement pour le hardware détecté."""
        if torch.cuda.is_available():
            # Détection des capacités GPU
            device_props = torch.cuda.get_device_properties(0)
            compute_capability = device_props.major * 10 + device_props.minor
            
            # Optimisations spécifiques Ampere (RTX 30XX+)
            if compute_capability >= 80:
                self.performance_config.use_flash_attention = True
                self.performance_config.allow_tf32 = True
                logger.info("Optimisations Ampere activées")
            
            # Optimisations spécifiques Ada Lovelace (RTX 40XX)
            if compute_capability >= 89:
                self.performance_config.precision_mode = PrecisionMode.FP16
                self.performance_config.tensorrt_optimization = True
                logger.info("Optimisations Ada Lovelace activées")
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'NeuroLiteConfig':
        """Charge la configuration depuis un fichier JSON."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'NeuroLiteConfig':
        """Crée la configuration depuis un dictionnaire."""
        # Conversion des enums
        if 'optimization_level' in config_dict:
            config_dict['optimization_level'] = OptimizationLevel(config_dict['optimization_level'])
        
        # Récursion pour les sous-configs
        for key, value in config_dict.items():
            if key.endswith('_config') and isinstance(value, dict):
                config_class_name = ''.join(word.capitalize() for word in key.split('_'))
                config_class = globals().get(config_class_name)
                if config_class:
                    config_dict[key] = config_class(**value)
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Exporte la configuration vers un dictionnaire."""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                result[key] = value.__dict__
            elif isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result
    
    def save_to_file(self, config_path: Union[str, Path]):
        """Sauvegarde la configuration dans un fichier JSON."""
        config_dict = self.to_dict()
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Configuration sauvegardée: {config_path}")
    
    def get_memory_requirements(self) -> Dict[str, float]:
        """Estime les besoins mémoire du modèle."""
        # Estimation paramètres modèle
        model_params = (
            self.model_config.hidden_size * self.model_config.hidden_size * self.model_config.num_layers * 8 +  # Attention
            self.model_config.hidden_size * self.model_config.intermediate_size * self.model_config.num_layers * 2  # FFN
        )
        
        # Facteur de précision
        precision_factor = {
            PrecisionMode.FP32: 4,
            PrecisionMode.FP16: 2,
            PrecisionMode.BF16: 2,
            PrecisionMode.INT8: 1
        }[self.performance_config.precision_mode]
        
        model_memory_gb = model_params * precision_factor / 1e9
        
        # Mémoire des activations (estimation)
        activation_memory_gb = (
            self.model_config.max_batch_size * 
            self.model_config.max_sequence_length * 
            self.model_config.hidden_size * 
            self.model_config.num_layers * 
            precision_factor / 1e9
        )
        
        # Mémoire des modules spécialisés
        memory_systems_gb = self.model_config.memory_capacity_mb / 1024
        
        return {
            'model_parameters': model_memory_gb,
            'activations': activation_memory_gb,
            'memory_systems': memory_systems_gb,
            'total_estimated': model_memory_gb + activation_memory_gb + memory_systems_gb
        }
    
    def optimize_for_environment(self, target_latency_ms: Optional[float] = None, 
                                target_memory_gb: Optional[float] = None):
        """Optimise automatiquement pour des contraintes de latence/mémoire."""
        if target_memory_gb:
            current_memory = self.get_memory_requirements()['total_estimated']
            if current_memory > target_memory_gb:
                # Réduction progressive des capacités
                reduction_factor = target_memory_gb / current_memory
                
                self.model_config.hidden_size = int(self.model_config.hidden_size * reduction_factor ** 0.5)
                self.model_config.num_layers = int(self.model_config.num_layers * reduction_factor ** 0.5)
                self.model_config.memory_capacity_mb = int(self.model_config.memory_capacity_mb * reduction_factor)
                
                logger.info(f"Configuration optimisée pour {target_memory_gb}GB mémoire")
        
        if target_latency_ms:
            if target_latency_ms < 100:  # Latence très faible
                self.optimization_level = OptimizationLevel.INFERENCE
                self.performance_config.quantization_enabled = True
                self.consciousness_config.introspection_frequency = 0.01
                logger.info("Optimisation ultra-basse latence activée")

# Configuration par défaut optimisée
def create_default_config() -> NeuroLiteConfig:
    """Crée une configuration par défaut optimisée."""
    return NeuroLiteConfig(
        optimization_level=OptimizationLevel.PRODUCTION,
        environment="production"
    )

def create_development_config() -> NeuroLiteConfig:
    """Crée une configuration de développement."""
    config = NeuroLiteConfig(
        optimization_level=OptimizationLevel.DEVELOPMENT,
        environment="development"
    )
    # Réductions pour le développement
    config.model_config.hidden_size = 512
    config.model_config.num_layers = 8
    config.performance_config.enable_profiling = True
    return config

def create_edge_config() -> NeuroLiteConfig:
    """Crée une configuration optimisée pour edge computing."""
    return NeuroLiteConfig(
        optimization_level=OptimizationLevel.EDGE,
        environment="edge"
    )

def create_tiny_config() -> NeuroLiteConfig:
    """Crée une configuration ultra-compacte pour tests et prototypage."""
    config = NeuroLiteConfig(
        optimization_level=OptimizationLevel.DEVELOPMENT,
        environment="development"
    )
    
    # Modèle ultra-compact
    config.model_config.hidden_size = 128
    config.model_config.num_layers = 2
    config.model_config.num_attention_heads = 2
    config.model_config.intermediate_size = 512
    
    # SSM minimal
    config.model_config.ssm_state_size = 4
    config.model_config.ssm_expand_factor = 1
    
    # Conscience basique
    config.model_config.consciousness_levels = 2
    config.model_config.consciousness_hidden_size = 64
    config.model_config.introspection_depth = 1
    
    # Mémoire minimale
    config.model_config.memory_capacity_mb = 128
    config.model_config.working_memory_size = 8
    config.model_config.episodic_memory_retention_hours = 12
    
    # World model basique
    config.model_config.world_simulation_steps = 3
    config.model_config.physics_simulation_enabled = False
    config.model_config.environment_complexity = "low"
    
    # Contraintes très réduites
    config.model_config.max_sequence_length = 512
    config.model_config.max_batch_size = 4
    config.model_config.max_memory_gb = 2.0
    
    # Modules mémoire minimaux
    config.memory_config.working_memory_capacity = 3
    config.memory_config.episodic_memory_mb = 64
    config.memory_config.semantic_memory_mb = 128
    config.memory_config.procedural_memory_mb = 32
    
    # Raisonnement minimal
    config.reasoning_config.enabled_types = ["deductive", "inductive"]
    config.reasoning_config.max_reasoning_steps = 3
    
    # Conscience ultra-minimaliste
    config.consciousness_config.max_level = "aware"
    config.consciousness_config.introspection_frequency = 0.02
    config.consciousness_config.self_reflection_depth = 1
    config.consciousness_config.metacognition_enabled = False
    
    # Planification minimale
    config.planning_config.enable_strategic = False
    config.planning_config.enable_tactical = False
    config.planning_config.enable_operational = True
    config.planning_config.short_term_horizon = 3
    config.planning_config.medium_term_horizon = 8
    config.planning_config.long_term_horizon = 20
    
    # Performance ultra-optimisée
    config.performance_config.precision_mode = PrecisionMode.INT8
    config.performance_config.quantization_enabled = True
    config.performance_config.cpu_offload = True
    config.performance_config.enable_profiling = False
    
    return config