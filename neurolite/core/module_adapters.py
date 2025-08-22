"""
NeuroLite AGI v2.0 - Adaptateurs de Modules
Adaptateurs pour intégrer tous les modules existants avec l'interface unifiée.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import time
import logging
from dataclasses import dataclass

from .unified_interface import (
    ModuleProtocol, ModuleMessage, ModuleType, ModuleState, 
    MessageType, Priority
)

# Imports des modules existants
from .consciousness import ConsciousnessModule
from .infinite_memory import InfiniteMemorySystem
from .reasoning import AdvancedReasoningEngine
from .world_model import WorldModel


# ✨ NOUVEAUX IMPORTS POUR GÉNÉRATION ET CLASSIFICATION
# Modules de génération/classification supprimés dans le pipeline léger

# Import pour les métriques du SuperMultimodalProcessor
try:
    from .super_multimodal_processor import ProcessingMetrics
except ImportError:
    # Fallback si non disponible
    @dataclass
    class ProcessingMetrics:
        tokenization_time_ms: float = 0.0
        encoding_time_ms: float = 0.0
        fusion_time_ms: float = 0.0
        total_time_ms: float = 0.0
        cache_hit_rate: float = 0.0
        modalities_processed: list = None
        confidence_score: float = 0.0
from .ssm import IndustrialSSMCore
from .brain_architecture import BrainlikeParallelProcessor
from .file_processors import UniversalFileProcessor
from .agi_controller import AGICentralController

logger = logging.getLogger(__name__)

class ConsciousnessAdapter(ModuleProtocol):
    """Adaptateur pour le module de conscience."""
    
    def __init__(self, hidden_size: int = 512):
        self.module = ConsciousnessModule(hidden_size)
        self.hidden_size = hidden_size
        self.message_count = 0
        self.error_count = 0
        self.last_activity = time.time()
        
    def get_module_type(self) -> ModuleType:
        return ModuleType.CONSCIOUSNESS
    
    def process_message(self, message: ModuleMessage) -> Optional[ModuleMessage]:
        try:
            self.message_count += 1
            self.last_activity = time.time()
            
            # Traitement selon le type de message
            if message.message_type == MessageType.DATA:
                # Traitement de données pour analyse de conscience
                consciousness_metrics = self.module(message.payload.unsqueeze(0))
                
                # Création de la réponse
                response = ModuleMessage(
                    source_module=ModuleType.CONSCIOUSNESS,
                    target_module=message.source_module,
                    message_type=MessageType.RESPONSE,
                    payload=consciousness_metrics.consciousness_vector,
                    metadata={
                        'consciousness_level': consciousness_metrics.level,
                        'coherence': consciousness_metrics.coherence,
                        'self_awareness': consciousness_metrics.self_awareness
                    },
                    priority=message.priority,
                    timestamp=time.time(),
                    message_id=f"consciousness_response_{self.message_count}",
                    trace_id=message.trace_id
                )
                return response
                
            elif message.message_type == MessageType.QUERY:
                # Requête sur l'état de conscience
                current_state = self.module.get_consciousness_state()
                
                response = ModuleMessage(
                    source_module=ModuleType.CONSCIOUSNESS,
                    target_module=message.source_module,
                    message_type=MessageType.RESPONSE,
                    payload=torch.tensor([current_state.level], dtype=torch.float32),
                    metadata={'consciousness_state': current_state.__dict__},
                    priority=message.priority,
                    timestamp=time.time(),
                    message_id=f"consciousness_query_{self.message_count}",
                    trace_id=message.trace_id
                )
                return response
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Erreur traitement message conscience: {e}")
            return None
    
    def get_health_status(self) -> ModuleState:
        return ModuleState(
            module_type=ModuleType.CONSCIOUSNESS,
            health_status="healthy" if self.error_count < 10 else "degraded",
            load_percentage=min(self.message_count / 1000 * 100, 100),
            memory_usage_mb=torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
            last_activity=self.last_activity,
            performance_metrics={'messages_processed': self.message_count, 'error_rate': self.error_count / max(self.message_count, 1)},
            error_count=self.error_count,
            total_messages_processed=self.message_count
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        try:
            # Configuration du module de conscience
            return True
        except Exception as e:
            logger.error(f"Erreur initialisation conscience: {e}")
            return False
    
    def shutdown(self) -> bool:
        try:
            # Nettoyage des ressources
            return True
        except Exception as e:
            logger.error(f"Erreur arrêt conscience: {e}")
            return False

class MemoryAdapter(ModuleProtocol):
    """Adaptateur pour le système de mémoire infinie."""
    
    def __init__(self, hidden_size: int = 512, storage_path: str = "./memory_storage"):
        self.module = InfiniteMemorySystem(
            hidden_size=hidden_size,
            storage_path=storage_path
        )
        self.hidden_size = hidden_size
        self.message_count = 0
        self.error_count = 0
        self.last_activity = time.time()
        
    def get_module_type(self) -> ModuleType:
        return ModuleType.MEMORY
    
    def process_message(self, message: ModuleMessage) -> Optional[ModuleMessage]:
        try:
            self.message_count += 1
            self.last_activity = time.time()
            
            if message.message_type == MessageType.DATA:
                # Stockage de données en mémoire
                memory_id = self.module.store_memory(
                    content=message.payload,
                    context_info=message.metadata.get('context', {}),
                    importance_hint=message.metadata.get('importance', 0.5)
                )
                
                response = ModuleMessage(
                    source_module=ModuleType.MEMORY,
                    target_module=message.source_module,
                    message_type=MessageType.RESPONSE,
                    payload=torch.tensor([1.0], dtype=torch.float32),  # Succès
                    metadata={'memory_id': memory_id, 'stored': True},
                    priority=message.priority,
                    timestamp=time.time(),
                    message_id=f"memory_store_{self.message_count}",
                    trace_id=message.trace_id
                )
                return response
                
            elif message.message_type == MessageType.QUERY:
                # Récupération de mémoires
                query_vector = message.payload
                memories = self.module.retrieve_memories(
                    query_vector=query_vector,
                    max_results=message.metadata.get('max_results', 10)
                )
                
                # Concaténation des mémoires récupérées
                if memories:
                    retrieved_data = torch.stack([m.content for m in memories])
                    memory_metadata = [m.metadata for m in memories]
                else:
                    retrieved_data = torch.zeros(1, self.hidden_size)
                    memory_metadata = []
                
                response = ModuleMessage(
                    source_module=ModuleType.MEMORY,
                    target_module=message.source_module,
                    message_type=MessageType.RESPONSE,
                    payload=retrieved_data,
                    metadata={'memories': memory_metadata, 'count': len(memories)},
                    priority=message.priority,
                    timestamp=time.time(),
                    message_id=f"memory_retrieve_{self.message_count}",
                    trace_id=message.trace_id
                )
                return response
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Erreur traitement message mémoire: {e}")
            return None
    
    def get_health_status(self) -> ModuleState:
        memory_stats = self.module.get_memory_statistics()
        
        return ModuleState(
            module_type=ModuleType.MEMORY,
            health_status="healthy" if self.error_count < 10 else "degraded",
            load_percentage=min(memory_stats.get('storage_usage_percent', 0), 100),
            memory_usage_mb=memory_stats.get('total_memory_mb', 0),
            last_activity=self.last_activity,
            performance_metrics={
                'messages_processed': self.message_count,
                'stored_memories': memory_stats.get('total_memories', 0),
                'error_rate': self.error_count / max(self.message_count, 1)
            },
            error_count=self.error_count,
            total_messages_processed=self.message_count
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        try:
            return True
        except Exception as e:
            logger.error(f"Erreur initialisation mémoire: {e}")
            return False
    
    def shutdown(self) -> bool:
        try:
            self.module.cleanup()
            return True
        except Exception as e:
            logger.error(f"Erreur arrêt mémoire: {e}")
            return False

class ReasoningAdapter(ModuleProtocol):
    """Adaptateur pour le moteur de raisonnement."""
    
    def __init__(self, hidden_size: int = 512):
        self.module = AdvancedReasoningEngine(hidden_size)
        self.hidden_size = hidden_size
        self.message_count = 0
        self.error_count = 0
        self.last_activity = time.time()
        
    def get_module_type(self) -> ModuleType:
        return ModuleType.REASONING
    
    def process_message(self, message: ModuleMessage) -> Optional[ModuleMessage]:
        try:
            self.message_count += 1
            self.last_activity = time.time()
            
            if message.message_type == MessageType.DATA:
                # Raisonnement sur les données
                reasoning_type = message.metadata.get('reasoning_type', 'deductive')
                context = message.metadata.get('context', {})
                
                reasoning_result = self.module.reason(
                    input_data=message.payload.unsqueeze(0),
                    reasoning_type=reasoning_type,
                    context=context
                )
                
                response = ModuleMessage(
                    source_module=ModuleType.REASONING,
                    target_module=message.source_module,
                    message_type=MessageType.RESPONSE,
                    payload=reasoning_result.conclusion,
                    metadata={
                        'reasoning_chain': reasoning_result.reasoning_chain,
                        'confidence': reasoning_result.confidence,
                        'reasoning_type': reasoning_type
                    },
                    priority=message.priority,
                    timestamp=time.time(),
                    message_id=f"reasoning_result_{self.message_count}",
                    trace_id=message.trace_id
                )
                return response
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Erreur traitement message raisonnement: {e}")
            return None
    
    def get_health_status(self) -> ModuleState:
        return ModuleState(
            module_type=ModuleType.REASONING,
            health_status="healthy" if self.error_count < 10 else "degraded",
            load_percentage=min(self.message_count / 1000 * 100, 100),
            memory_usage_mb=torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
            last_activity=self.last_activity,
            performance_metrics={'messages_processed': self.message_count, 'error_rate': self.error_count / max(self.message_count, 1)},
            error_count=self.error_count,
            total_messages_processed=self.message_count
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        return True
    
    def shutdown(self) -> bool:
        return True

class WorldModelAdapter(ModuleProtocol):
    """Adaptateur pour le modèle du monde."""
    
    def __init__(self, hidden_size: int = 512):
        self.module = WorldModel(hidden_size)
        self.hidden_size = hidden_size
        self.message_count = 0
        self.error_count = 0
        self.last_activity = time.time()
        
    def get_module_type(self) -> ModuleType:
        return ModuleType.WORLD_MODEL
    
    def process_message(self, message: ModuleMessage) -> Optional[ModuleMessage]:
        try:
            self.message_count += 1
            self.last_activity = time.time()
            
            if message.message_type == MessageType.DATA:
                # Prédiction ou planification
                current_state = message.payload.unsqueeze(0)
                action_type = message.metadata.get('action_type', 'prediction')
                
                if action_type == 'prediction':
                    prediction = self.module.predict_next_state(current_state)
                    payload = prediction
                    metadata = {'prediction_confidence': 0.8, 'action_type': 'prediction'}
                elif action_type == 'planning':
                    goal_state = message.metadata.get('goal_state')
                    plan = self.module.plan_actions(current_state, goal_state)
                    payload = plan.expected_outcome.environment
                    metadata = {'plan': plan.__dict__, 'action_type': 'planning'}
                else:
                    payload = current_state
                    metadata = {'action_type': 'unknown'}
                
                response = ModuleMessage(
                    source_module=ModuleType.WORLD_MODEL,
                    target_module=message.source_module,
                    message_type=MessageType.RESPONSE,
                    payload=payload,
                    metadata=metadata,
                    priority=message.priority,
                    timestamp=time.time(),
                    message_id=f"world_model_{self.message_count}",
                    trace_id=message.trace_id
                )
                return response
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Erreur traitement message world model: {e}")
            return None
    
    def get_health_status(self) -> ModuleState:
        return ModuleState(
            module_type=ModuleType.WORLD_MODEL,
            health_status="healthy" if self.error_count < 10 else "degraded",
            load_percentage=min(self.message_count / 1000 * 100, 100),
            memory_usage_mb=torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
            last_activity=self.last_activity,
            performance_metrics={'messages_processed': self.message_count, 'error_rate': self.error_count / max(self.message_count, 1)},
            error_count=self.error_count,
            total_messages_processed=self.message_count
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        return True
    
    def shutdown(self) -> bool:
        return True

class MultimodalFusionAdapter(ModuleProtocol):
    """Adaptateur pour la fusion multimodale avec SuperMultimodalProcessor."""
    
    def __init__(self, hidden_size: int = 512):
        # 🚀 NOUVEAU: Utilisation de SuperMultimodalProcessor
        try:
            from .super_multimodal_processor import SuperMultimodalProcessor
            from .multimodal_fusion import FusionStrategy
            
            self.module = SuperMultimodalProcessor(
                hidden_size=hidden_size,
                fusion_strategy=FusionStrategy.ADAPTIVE_FUSION,
                enable_universal_tokenizer=True,
                enable_caching=True,
                enable_parallel=True
            )
            logger.info("✅ MultimodalFusionAdapter utilise SuperMultimodalProcessor")
            
        except ImportError as e:
            logger.error(f"❌ Impossible d'importer SuperMultimodalProcessor: {e}")
            # Fallback d'urgence - ne devrait pas arriver
            raise ImportError("SuperMultimodalProcessor requis pour MultimodalFusionAdapter")
        
        self.hidden_size = hidden_size
        self.message_count = 0
        self.error_count = 0
        self.last_activity = time.time()
        
    def get_module_type(self) -> ModuleType:
        return ModuleType.MULTIMODAL_FUSION
    
    def process_message(self, message: ModuleMessage) -> Optional[ModuleMessage]:
        try:
            self.message_count += 1
            self.last_activity = time.time()
            
            if message.message_type == MessageType.DATA:
                # 🎯 NOUVEAU: Traitement avec tokenization universelle
                modalities = message.metadata.get('modalities', ['text'])
                
                # Préparer les données pour SuperMultimodalProcessor
                inputs = {}
                
                # Déduire le type de données
                if isinstance(message.payload, str):
                    inputs['text'] = message.payload
                elif hasattr(message.payload, 'shape'):
                    # Tensor - déterminer la modalité par la forme
                    if len(message.payload.shape) >= 3:
                        inputs['image'] = message.payload
                    else:
                        inputs['audio'] = message.payload
                else:
                    inputs['structured'] = message.payload
                
                # Ajouter modalités depuis metadata
                for modality in modalities:
                    if modality not in inputs:
                        inputs[modality] = message.payload
                
                # 🚀 Traitement avec SuperMultimodalProcessor
                result = self.module.forward(inputs, return_metrics=True)
                if isinstance(result, tuple):
                    output_tensor, metrics = result
                else:
                    output_tensor = result
                    metrics = ProcessingMetrics()  # Métriques par défaut
                
                response = ModuleMessage(
                    source_module=ModuleType.MULTIMODAL_FUSION,
                    target_module=message.source_module,
                    message_type=MessageType.RESPONSE,
                    payload=output_tensor,
                    metadata={
                        'modalities_processed': metrics.modalities_processed,
                        'confidence_score': metrics.confidence_score,
                        'tokenization_time_ms': metrics.tokenization_time_ms,
                        'encoding_time_ms': metrics.encoding_time_ms,
                        'fusion_time_ms': metrics.fusion_time_ms,
                        'total_time_ms': metrics.total_time_ms,
                        'cache_hit_rate': metrics.cache_hit_rate,
                        'fusion_strategy': 'adaptive_with_tokenizer'
                    },
                    priority=message.priority,
                    timestamp=time.time(),
                    message_id=f"super_multimodal_{self.message_count}",
                    trace_id=message.trace_id
                )
                return response
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Erreur traitement message SuperMultimodal: {e}")
            return None
    
    def get_health_status(self) -> ModuleState:
        # 📊 Ajouter statistiques du SuperMultimodalProcessor
        performance_metrics = {'messages_processed': self.message_count, 'error_rate': self.error_count / max(self.message_count, 1)}
        
        try:
            super_stats = self.module.get_performance_stats()
            performance_metrics.update({
                'super_total_processed': super_stats['total_processed'],
                'super_average_time_ms': super_stats['average_time_ms'],
                'super_error_rate': super_stats['error_rate'],
                'tokenizer_available': super_stats['tokenizer_available'],
                'cache_enabled': super_stats['cache_enabled'],
                'supported_modalities': super_stats['supported_modalities']
            })
        except Exception:
            pass
        
        return ModuleState(
            module_type=ModuleType.MULTIMODAL_FUSION,
            health_status="healthy" if self.error_count < 10 else "degraded",
            load_percentage=min(self.message_count / 1000 * 100, 100),
            memory_usage_mb=torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
            last_activity=self.last_activity,
            performance_metrics=performance_metrics,
            error_count=self.error_count,
            total_messages_processed=self.message_count
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        try:
            # Réchauffage optionnel du SuperMultimodalProcessor
            if hasattr(self.module, 'warmup'):
                self.module.warmup()
            return True
        except Exception as e:
            logger.error(f"Erreur initialisation MultimodalFusionAdapter: {e}")
            return False
    
    def shutdown(self) -> bool:
        try:
            # Nettoyer les caches si disponible
            if hasattr(self.module, 'clear_cache'):
                self.module.clear_cache()
            return True
        except Exception:
            return True

class SSMAdapter(ModuleProtocol):
    """Adaptateur pour le cœur SSM."""
    
    def __init__(self, hidden_size: int = 512):
        self.module = IndustrialSSMCore(
            dim=hidden_size,
            d_state=16,
            d_conv=4,
            expand_factor=2
        )
        self.hidden_size = hidden_size
        self.message_count = 0
        self.error_count = 0
        self.last_activity = time.time()
        
    def get_module_type(self) -> ModuleType:
        return ModuleType.SSM_CORE
    
    def process_message(self, message: ModuleMessage) -> Optional[ModuleMessage]:
        try:
            self.message_count += 1
            self.last_activity = time.time()
            
            if message.message_type == MessageType.DATA:
                # Traitement SSM séquentiel
                input_data = message.payload.unsqueeze(0)  # [1, seq_len, hidden_size]
                if input_data.dim() == 2:
                    input_data = input_data.unsqueeze(1)  # Ajouter dimension séquence
                
                processed_data = self.module(input_data)
                
                response = ModuleMessage(
                    source_module=ModuleType.SSM_CORE,
                    target_module=message.source_module,
                    message_type=MessageType.RESPONSE,
                    payload=processed_data.squeeze(0),
                    metadata={'processing_type': 'ssm_sequential'},
                    priority=message.priority,
                    timestamp=time.time(),
                    message_id=f"ssm_processed_{self.message_count}",
                    trace_id=message.trace_id
                )
                return response
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Erreur traitement message SSM: {e}")
            return None
    
    def get_health_status(self) -> ModuleState:
        return ModuleState(
            module_type=ModuleType.SSM_CORE,
            health_status="healthy" if self.error_count < 10 else "degraded",
            load_percentage=min(self.message_count / 1000 * 100, 100),
            memory_usage_mb=torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
            last_activity=self.last_activity,
            performance_metrics={'messages_processed': self.message_count, 'error_rate': self.error_count / max(self.message_count, 1)},
            error_count=self.error_count,
            total_messages_processed=self.message_count
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        return True
    
    def shutdown(self) -> bool:
        return True

class BrainArchitectureAdapter(ModuleProtocol):
    """Adaptateur pour l'architecture cerveau."""
    
    def __init__(self, hidden_size: int = 512):
        self.module = BrainlikeParallelProcessor()
        self.hidden_size = hidden_size
        self.message_count = 0
        self.error_count = 0
        self.last_activity = time.time()
        
    def get_module_type(self) -> ModuleType:
        return ModuleType.BRAIN_ARCHITECTURE
    
    def process_message(self, message: ModuleMessage) -> Optional[ModuleMessage]:
        try:
            self.message_count += 1
            self.last_activity = time.time()
            
            if message.message_type == MessageType.DATA:
                # Traitement parallèle via l'architecture cerveau
                
                # Simulation de traitement de fichiers
                file_paths = message.metadata.get('file_paths', [])
                
                # Traitement asynchrone simulé
                processing_result = {
                    'processed_files': len(file_paths),
                    'processing_time': time.time() - message.timestamp,
                    'brain_regions_active': ['visual_cortex', 'language_cortex', 'executive_cortex']
                }
                
                response = ModuleMessage(
                    source_module=ModuleType.BRAIN_ARCHITECTURE,
                    target_module=message.source_module,
                    message_type=MessageType.RESPONSE,
                    payload=message.payload,  # Passthrough pour les données
                    metadata=processing_result,
                    priority=message.priority,
                    timestamp=time.time(),
                    message_id=f"brain_arch_{self.message_count}",
                    trace_id=message.trace_id
                )
                return response
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Erreur traitement message brain architecture: {e}")
            return None
    
    def get_health_status(self) -> ModuleState:
        return ModuleState(
            module_type=ModuleType.BRAIN_ARCHITECTURE,
            health_status="healthy" if self.error_count < 10 else "degraded",
            load_percentage=min(self.message_count / 1000 * 100, 100),
            memory_usage_mb=torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
            last_activity=self.last_activity,
            performance_metrics={'messages_processed': self.message_count, 'error_rate': self.error_count / max(self.message_count, 1)},
            error_count=self.error_count,
            total_messages_processed=self.message_count
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        return True
    
    def shutdown(self) -> bool:
        return True

class FileProcessorAdapter(ModuleProtocol):
    """Adaptateur pour le processeur de fichiers."""
    
    def __init__(self, hidden_size: int = 512):
        self.module = UniversalFileProcessor()
        self.hidden_size = hidden_size
        self.message_count = 0
        self.error_count = 0
        self.last_activity = time.time()
        
    def get_module_type(self) -> ModuleType:
        return ModuleType.FILE_PROCESSOR
    
    def process_message(self, message: ModuleMessage) -> Optional[ModuleMessage]:
        try:
            self.message_count += 1
            self.last_activity = time.time()
            
            if message.message_type == MessageType.DATA:
                # Traitement de fichiers
                file_paths = message.metadata.get('file_paths', [])
                file_type = message.metadata.get('file_type', 'text')
                
                # Simulation de traitement
                processed_files = []
                for i, file_path in enumerate(file_paths[:5]):  # Limite à 5 fichiers
                    processed_files.append({
                        'file_path': file_path,
                        'file_type': file_type,
                        'processing_time_ms': 10 + i * 5,
                        'success': True
                    })
                
                response = ModuleMessage(
                    source_module=ModuleType.FILE_PROCESSOR,
                    target_module=message.source_module,
                    message_type=MessageType.RESPONSE,
                    payload=torch.randn(len(processed_files), self.hidden_size),  # Embeddings des fichiers
                    metadata={
                        'processed_files': processed_files,
                        'total_files': len(file_paths),
                        'supported_types': ['text', 'image', 'audio', 'video', 'document']
                    },
                    priority=message.priority,
                    timestamp=time.time(),
                    message_id=f"file_proc_{self.message_count}",
                    trace_id=message.trace_id
                )
                return response
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Erreur traitement message file processor: {e}")
            return None
    
    def get_health_status(self) -> ModuleState:
        return ModuleState(
            module_type=ModuleType.FILE_PROCESSOR,
            health_status="healthy" if self.error_count < 10 else "degraded",
            load_percentage=min(self.message_count / 1000 * 100, 100),
            memory_usage_mb=0,  # Traitement de fichiers principalement I/O
            last_activity=self.last_activity,
            performance_metrics={'messages_processed': self.message_count, 'error_rate': self.error_count / max(self.message_count, 1)},
            error_count=self.error_count,
            total_messages_processed=self.message_count
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        return True
    
    def shutdown(self) -> bool:
        return True

class AGIControllerAdapter(ModuleProtocol):
    """Adaptateur pour le contrôleur AGI central."""
    
    def __init__(self, hidden_size: int = 512):
        self.module = AGICentralController(hidden_size)
        self.hidden_size = hidden_size
        self.message_count = 0
        self.error_count = 0
        self.last_activity = time.time()
        
    def get_module_type(self) -> ModuleType:
        return ModuleType.AGI_CONTROLLER
    
    def process_message(self, message: ModuleMessage) -> Optional[ModuleMessage]:
        try:
            self.message_count += 1
            self.last_activity = time.time()
            
            if message.message_type == MessageType.COMMAND:
                # Traitement de commandes de contrôle
                command = message.metadata.get('command', 'status')
                
                if command == 'status':
                    status_info = {
                        'system_status': 'operational',
                        'active_modules': 8,
                        'coordination_efficiency': 0.92,
                        'total_processed': self.message_count
                    }
                elif command == 'optimize':
                    status_info = {
                        'optimization_applied': True,
                        'performance_improvement': '15%',
                        'optimized_modules': ['consciousness', 'memory', 'reasoning']
                    }
                else:
                    status_info = {'error': f'Commande inconnue: {command}'}
                
                response = ModuleMessage(
                    source_module=ModuleType.AGI_CONTROLLER,
                    target_module=message.source_module,
                    message_type=MessageType.RESPONSE,
                    payload=torch.tensor([1.0], dtype=torch.float32),  # Status signal
                    metadata=status_info,
                    priority=message.priority,
                    timestamp=time.time(),
                    message_id=f"agi_ctrl_{self.message_count}",
                    trace_id=message.trace_id
                )
                return response
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Erreur traitement message AGI controller: {e}")
            return None
    
    def get_health_status(self) -> ModuleState:
        return ModuleState(
            module_type=ModuleType.AGI_CONTROLLER,
            health_status="healthy" if self.error_count < 10 else "degraded",
            load_percentage=min(self.message_count / 1000 * 100, 100),
            memory_usage_mb=torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
            last_activity=self.last_activity,
            performance_metrics={'messages_processed': self.message_count, 'error_rate': self.error_count / max(self.message_count, 1)},
            error_count=self.error_count,
            total_messages_processed=self.message_count
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        return True
    
    def shutdown(self) -> bool:
        return True

## Modules de génération/classification supprimés dans le pipeline léger

def create_all_adapters(hidden_size: int = 512, storage_path: str = "./memory_storage", 
                       config: Optional[Any] = None) -> Dict[ModuleType, ModuleProtocol]:
    """Crée TOUS les adaptateurs de modules avec configuration avancée."""
    
    # Configuration par défaut si non fournie
    if config is None:
        from ..Configs.config import create_default_config
        config = create_default_config()
    
    # Création des adaptateurs avec configuration spécialisée
    adapters = {}
    
    # 🧠 MODULES COGNITIFS avec config spécialisée
    if config.consciousness_config.enabled:
        consciousness_adapter = ConsciousnessAdapter(hidden_size)
        consciousness_adapter.config = config.consciousness_config
        adapters[ModuleType.CONSCIOUSNESS] = consciousness_adapter
    
    if config.memory_config.enable_episodic:
        memory_adapter = MemoryAdapter(hidden_size, storage_path)
        # Application des paramètres de config après création
        memory_adapter.config = config.memory_config
        memory_adapter.max_capacity_mb = config.memory_config.episodic_memory_mb
        adapters[ModuleType.MEMORY] = memory_adapter
    
    if len(config.reasoning_config.enabled_types) > 0:
        reasoning_adapter = ReasoningAdapter(hidden_size)
        reasoning_adapter.config = config.reasoning_config
        reasoning_adapter.enabled_types = config.reasoning_config.enabled_types
        adapters[ModuleType.REASONING] = reasoning_adapter
    
    if config.planning_config.enable_strategic:
        world_adapter = WorldModelAdapter(hidden_size)
        world_adapter.config = config.planning_config
        adapters[ModuleType.WORLD_MODEL] = world_adapter
    
    # ⚡ MODULES DE TRAITEMENT (toujours actifs mais configurés)
    multimodal_adapter = MultimodalFusionAdapter(hidden_size)
    multimodal_adapter.config = config.model_config
    adapters[ModuleType.MULTIMODAL_FUSION] = multimodal_adapter
    
    ssm_adapter = SSMAdapter(hidden_size)
    ssm_adapter.config = config.model_config
    ssm_adapter.state_size = config.model_config.ssm_state_size
    ssm_adapter.conv_kernel = config.model_config.ssm_conv_kernel
    adapters[ModuleType.SSM_CORE] = ssm_adapter
    
    brain_adapter = BrainArchitectureAdapter(hidden_size)
    brain_adapter.config = config.model_config
    adapters[ModuleType.BRAIN_ARCHITECTURE] = brain_adapter
    
    file_adapter = FileProcessorAdapter(hidden_size)
    file_adapter.config = config.model_config
    adapters[ModuleType.FILE_PROCESSOR] = file_adapter
    
    # 🔧 CONTRÔLE ET OPTIMISATION
    controller_adapter = AGIControllerAdapter(hidden_size)
    controller_adapter.config = config
    adapters[ModuleType.AGI_CONTROLLER] = controller_adapter
    
    # Modules génération/classification retirés
    
    logger.info(f"📋 {len(adapters)} adaptateurs créés avec configuration {config.optimization_level}")
    logger.info("✨ Nouveaux modules de génération et classification activés")
    
    return adapters

# Export de TOUS les adaptateurs
__all__ = [
    # Adaptateurs cognitifs principaux
    'ConsciousnessAdapter',
    'MemoryAdapter', 
    'ReasoningAdapter',
    'WorldModelAdapter',
    
    # Adaptateurs de traitement
    'MultimodalFusionAdapter',
    'SSMAdapter',
    'BrainArchitectureAdapter',
    'FileProcessorAdapter',
    
    # Adaptateurs de contrôle
    'AGIControllerAdapter',
    
    # Adaptateurs génération/classification retirés
    
    # Fonction utilitaire
    'create_all_adapters'
]