"""
NeuroLite AGI - Architecture Cerveau avec Traitement Parallèle
Simule les régions cérébrales spécialisées avec traitement concurrent
"""

import asyncio
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
from collections import defaultdict, deque
import json

from .file_processors import ProcessedFile, FileType, UniversalFileProcessor

class BrainRegion(Enum):
    """Régions cérébrales spécialisées"""
    VISUAL_CORTEX = "visual_cortex"          # Images, vidéos
    AUDITORY_CORTEX = "auditory_cortex"      # Audio, sons
    LANGUAGE_CORTEX = "language_cortex"      # Texte, langage
    ANALYTICAL_CORTEX = "analytical_cortex"  # Données, analyse
    WEB_CORTEX = "web_cortex"               # Contenu web
    MOTOR_CORTEX = "motor_cortex"           # Actions, commandes
    MEMORY_CORTEX = "memory_cortex"         # Stockage, récupération
    EXECUTIVE_CORTEX = "executive_cortex"   # Contrôle, décision

@dataclass
class BrainSignal:
    """Signal nerveux entre régions cérébrales"""
    source_region: BrainRegion
    target_region: BrainRegion
    content: torch.Tensor
    metadata: Dict[str, Any]
    timestamp: float
    priority: int = 1  # 1=low, 5=critical
    signal_type: str = "data"

@dataclass
class ProcessingTask:
    """Tâche de traitement pour une région cérébrale"""
    task_id: str
    region: BrainRegion
    files: List[str]
    callback: Optional[Callable] = None
    priority: int = 1
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

class NeuralPathway:
    """Voie neurale pour communication inter-régions"""
    
    def __init__(self, source: BrainRegion, target: BrainRegion, strength: float = 1.0):
        self.source = source
        self.target = target
        self.strength = strength
        self.signal_history = deque(maxlen=100)
        self.usage_count = 0
        self.last_used = 0.0
        
        # Réseau neuronal pour transformation du signal
        self.transform_network = nn.Sequential(
            nn.Linear(512, 256),  # Taille standard des signaux
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512)
        )
    
    def transmit_signal(self, signal: BrainSignal) -> BrainSignal:
        """Transmet et transforme un signal"""
        # Transformation neurale du signal
        with torch.no_grad():
            if signal.content.dim() == 1:
                signal.content = signal.content.unsqueeze(0)
            transformed_content = self.transform_network(signal.content)
        
        # Mise à jour des statistiques
        self.usage_count += 1
        self.last_used = time.time()
        self.signal_history.append({
            'timestamp': signal.timestamp,
            'signal_type': signal.signal_type,
            'strength': self.strength
        })
        
        # Créer le signal transformé
        transformed_signal = BrainSignal(
            source_region=self.source,
            target_region=self.target,
            content=transformed_content,
            metadata={**signal.metadata, 'pathway_strength': self.strength},
            timestamp=time.time(),
            priority=signal.priority,
            signal_type=signal.signal_type
        )
        
        return transformed_signal
    
    def strengthen(self, factor: float = 1.1):
        """Renforce la connexion (apprentissage hébbien)"""
        self.strength = min(2.0, self.strength * factor)
    
    def weaken(self, factor: float = 0.9):
        """Affaiblit la connexion"""
        self.strength = max(0.1, self.strength * factor)

class CorticalRegion:
    """Région corticale spécialisée avec traitement asynchrone"""
    
    def __init__(self, region_type: BrainRegion, processing_capacity: int = 4):
        self.region_type = region_type
        self.processing_capacity = processing_capacity
        self.is_active = True
        
        # Files d'attente
        self.input_queue = queue.PriorityQueue()
        self.output_queue = queue.Queue()
        
        # Processeur de fichiers
        self.file_processor = UniversalFileProcessor()
        
        # Thread pool pour traitement parallèle
        self.executor = ThreadPoolExecutor(max_workers=processing_capacity)
        
        # Statistiques
        self.processed_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        
        # Spécialisation par région
        self.specialized_file_types = self._get_specialized_types()
        
        # Réseau neuronal spécialisé pour cette région
        self.neural_processor = self._create_specialized_network()
        
        # Thread principal de traitement
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
    
    def _get_specialized_types(self) -> List[FileType]:
        """Retourne les types de fichiers spécialisés pour cette région"""
        specialization = {
            BrainRegion.VISUAL_CORTEX: [FileType.IMAGE, FileType.VIDEO],
            BrainRegion.AUDITORY_CORTEX: [FileType.AUDIO],
            BrainRegion.LANGUAGE_CORTEX: [FileType.TEXT, FileType.DOCUMENT],
            BrainRegion.ANALYTICAL_CORTEX: [FileType.DATA],
            BrainRegion.WEB_CORTEX: [FileType.WEB],
            BrainRegion.MOTOR_CORTEX: [FileType.CODE],
        }
        return specialization.get(self.region_type, [])
    
    def _create_specialized_network(self) -> nn.Module:
        """Crée un réseau neuronal spécialisé pour cette région"""
        if self.region_type == BrainRegion.VISUAL_CORTEX:
            # Réseau type CNN pour vision
            return nn.Sequential(
                nn.Conv1d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(512),
                nn.Flatten(),
                nn.Linear(512*32, 512),
                nn.LayerNorm(512)
            )
        elif self.region_type == BrainRegion.AUDITORY_CORTEX:
            # Réseau spécialisé audio
            return nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 512),
                nn.LayerNorm(512)
            )
        elif self.region_type == BrainRegion.LANGUAGE_CORTEX:
            # Réseau type Transformer pour langage
            return nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(1024, 512),
                nn.LayerNorm(512)
            )
        else:
            # Réseau générique
            return nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.LayerNorm(512)
            )
    
    def _processing_loop(self):
        """Boucle principale de traitement de la région"""
        while self.is_active:
            try:
                # Attendre une tâche (avec timeout)
                priority, task = self.input_queue.get(timeout=1.0)
                
                # Traiter la tâche
                asyncio.run(self._process_task(task))
                
                self.input_queue.task_done()
                
            except queue.Empty:
                continue  # Pas de tâche, continuer
            except Exception as e:
                print(f"❌ Erreur région {self.region_type.value}: {e}")
                self.error_count += 1
    
    async def _process_task(self, task: ProcessingTask):
        """Traite une tâche assignée à cette région"""
        start_time = time.time()
        
        try:
            # Traitement des fichiers en parallèle
            processed_files = await self.file_processor.process_multiple_files(task.files)
            
            # Filtrer les fichiers adaptés à cette région
            region_files = []
            for pf in processed_files:
                if isinstance(pf, ProcessedFile):
                    if not self.specialized_file_types or pf.file_type in self.specialized_file_types:
                        region_files.append(pf)
            
            if region_files:
                # Traitement neuronal spécialisé
                processed_signals = []
                for pf in region_files:
                    # Assurer la dimension correcte
                    content = pf.content
                    if content.numel() != 512:
                        # Adapter la taille à 512
                        if content.numel() < 512:
                            content = torch.cat([content.flatten(), torch.zeros(512 - content.numel())])
                        else:
                            content = content.flatten()[:512]
                    
                    # Traitement par le réseau spécialisé
                    with torch.no_grad():
                        if self.region_type == BrainRegion.VISUAL_CORTEX and len(content.shape) == 1:
                            content = content.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                            processed_content = self.neural_processor(content)
                        else:
                            if content.dim() == 1:
                                content = content.unsqueeze(0)
                            processed_content = self.neural_processor(content)
                    
                    # Créer un signal cérébral
                    signal = BrainSignal(
                        source_region=self.region_type,
                        target_region=BrainRegion.EXECUTIVE_CORTEX,  # Vers le contrôle exécutif
                        content=processed_content.squeeze(0) if processed_content.dim() > 1 else processed_content,
                        metadata={
                            'file_path': pf.file_path,
                            'file_type': pf.file_type.value,
                            'processing_region': self.region_type.value,
                            'original_metadata': pf.metadata
                        },
                        timestamp=time.time(),
                        priority=task.priority
                    )
                    processed_signals.append(signal)
                
                # Envoyer les signaux vers la sortie
                for signal in processed_signals:
                    self.output_queue.put(signal)
                
                # Callback si défini
                if task.callback:
                    task.callback(processed_signals)
            
            # Statistiques
            processing_time = time.time() - start_time
            self.processed_count += 1
            self.total_processing_time += processing_time
            
            print(f"✅ {self.region_type.value}: traité {len(region_files)} fichier(s) en {processing_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Erreur traitement tâche {task.task_id} dans {self.region_type.value}: {e}")
            self.error_count += 1
    
    def submit_task(self, task: ProcessingTask):
        """Soumet une tâche à cette région"""
        # Priorité négative pour queue.PriorityQueue (plus petit = plus prioritaire)
        self.input_queue.put((-task.priority, task))
    
    def get_output_signals(self) -> List[BrainSignal]:
        """Récupère les signaux de sortie"""
        signals = []
        while not self.output_queue.empty():
            try:
                signals.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return signals
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de la région"""
        avg_time = self.total_processing_time / max(1, self.processed_count)
        return {
            'region_type': self.region_type.value,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'avg_processing_time': avg_time,
            'queue_size': self.input_queue.qsize(),
            'specialization': [ft.value for ft in self.specialized_file_types],
            'is_active': self.is_active
        }
    
    def shutdown(self):
        """Arrêt propre de la région"""
        self.is_active = False
        self.executor.shutdown(wait=True)

class NeuralNetwork:
    """Réseau neuronal inter-régions du cerveau"""
    
    def __init__(self):
        self.pathways: Dict[tuple, NeuralPathway] = {}
        self.signal_routing = defaultdict(list)
        self.global_signal_history = deque(maxlen=1000)
    
    def create_pathway(self, source: BrainRegion, target: BrainRegion, strength: float = 1.0):
        """Crée une voie neurale entre deux régions"""
        pathway_key = (source, target)
        self.pathways[pathway_key] = NeuralPathway(source, target, strength)
        self.signal_routing[source].append(target)
    
    def route_signal(self, signal: BrainSignal) -> List[BrainSignal]:
        """Route un signal vers les régions appropriées"""
        routed_signals = []
        
        # Trouver les voies de sortie
        targets = self.signal_routing.get(signal.source_region, [])
        
        for target in targets:
            pathway_key = (signal.source_region, target)
            if pathway_key in self.pathways:
                pathway = self.pathways[pathway_key]
                transformed_signal = pathway.transmit_signal(signal)
                transformed_signal.target_region = target
                routed_signals.append(transformed_signal)
                
                # Renforcement de la voie si signal de haute priorité
                if signal.priority >= 4:
                    pathway.strengthen()
        
        # Historique global
        self.global_signal_history.append({
            'timestamp': signal.timestamp,
            'source': signal.source_region.value,
            'targets': [s.target_region.value for s in routed_signals],
            'signal_type': signal.signal_type
        })
        
        return routed_signals
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Statistiques du réseau neuronal"""
        return {
            'total_pathways': len(self.pathways),
            'pathway_strengths': {
                f"{k[0].value}->{k[1].value}": v.strength 
                for k, v in self.pathways.items()
            },
            'signal_history_count': len(self.global_signal_history),
            'routing_table': {
                k.value: [t.value for t in v] 
                for k, v in self.signal_routing.items()
            }
        }

class BrainlikeParallelProcessor:
    """Processeur principal simulant un cerveau complet"""
    
    def __init__(self, regions_config: Optional[Dict[BrainRegion, int]] = None):
        self.is_active = True
        
        # Configuration par défaut des régions
        if regions_config is None:
            regions_config = {
                BrainRegion.VISUAL_CORTEX: 2,
                BrainRegion.AUDITORY_CORTEX: 2, 
                BrainRegion.LANGUAGE_CORTEX: 4,
                BrainRegion.ANALYTICAL_CORTEX: 3,
                BrainRegion.WEB_CORTEX: 2,
                BrainRegion.EXECUTIVE_CORTEX: 1
            }
        
        # Créer les régions cérébrales
        self.regions: Dict[BrainRegion, CorticalRegion] = {}
        for region_type, capacity in regions_config.items():
            self.regions[region_type] = CorticalRegion(region_type, capacity)
        
        # Créer le réseau neuronal
        self.neural_network = NeuralNetwork()
        self._setup_neural_pathways()
        
        # Système de routage intelligent
        self.task_router = self._create_task_router()
        
        # Thread de coordination générale
        self.coordination_thread = threading.Thread(target=self._coordination_loop, daemon=True)
        self.coordination_thread.start()
        
        # Statistiques globales
        self.total_tasks_processed = 0
        self.start_time = time.time()
    
    def _setup_neural_pathways(self):
        """Configure les connexions entre régions cérébrales"""
        # Connexions principales (comme dans un cerveau réel)
        connections = [
            # Vers le cortex exécutif
            (BrainRegion.VISUAL_CORTEX, BrainRegion.EXECUTIVE_CORTEX, 1.5),
            (BrainRegion.AUDITORY_CORTEX, BrainRegion.EXECUTIVE_CORTEX, 1.4),
            (BrainRegion.LANGUAGE_CORTEX, BrainRegion.EXECUTIVE_CORTEX, 1.8),
            (BrainRegion.ANALYTICAL_CORTEX, BrainRegion.EXECUTIVE_CORTEX, 1.6),
            
            # Connexions inter-corticales
            (BrainRegion.VISUAL_CORTEX, BrainRegion.LANGUAGE_CORTEX, 1.2),
            (BrainRegion.AUDITORY_CORTEX, BrainRegion.LANGUAGE_CORTEX, 1.3),
            (BrainRegion.LANGUAGE_CORTEX, BrainRegion.ANALYTICAL_CORTEX, 1.1),
            
            # Vers la mémoire
            (BrainRegion.EXECUTIVE_CORTEX, BrainRegion.MEMORY_CORTEX, 1.7),
        ]
        
        for source, target, strength in connections:
            if source in self.regions and target in self.regions:
                self.neural_network.create_pathway(source, target, strength)
    
    def _create_task_router(self) -> Dict[FileType, List[BrainRegion]]:
        """Crée la table de routage des tâches"""
        return {
            FileType.TEXT: [BrainRegion.LANGUAGE_CORTEX],
            FileType.IMAGE: [BrainRegion.VISUAL_CORTEX],
            FileType.AUDIO: [BrainRegion.AUDITORY_CORTEX],
            FileType.VIDEO: [BrainRegion.VISUAL_CORTEX, BrainRegion.AUDITORY_CORTEX],
            FileType.DOCUMENT: [BrainRegion.LANGUAGE_CORTEX, BrainRegion.ANALYTICAL_CORTEX],
            FileType.DATA: [BrainRegion.ANALYTICAL_CORTEX],
            FileType.WEB: [BrainRegion.WEB_CORTEX, BrainRegion.LANGUAGE_CORTEX],
            FileType.CODE: [BrainRegion.MOTOR_CORTEX, BrainRegion.LANGUAGE_CORTEX],
        }
    
    def _coordination_loop(self):
        """Boucle de coordination inter-régions"""
        while self.is_active:
            try:
                # Collecter les signaux de toutes les régions
                all_signals = []
                for region in self.regions.values():
                    signals = region.get_output_signals()
                    all_signals.extend(signals)
                
                # Router les signaux via le réseau neuronal
                for signal in all_signals:
                    routed_signals = self.neural_network.route_signal(signal)
                    
                    # Distribuer aux régions cibles
                    for routed_signal in routed_signals:
                        target_region = self.regions.get(routed_signal.target_region)
                        if target_region:
                            # Créer une tâche interne pour le signal
                            internal_task = ProcessingTask(
                                task_id=f"internal_{int(time.time()*1000)}",
                                region=routed_signal.target_region,
                                files=[],  # Pas de fichiers, traitement de signal
                                priority=routed_signal.priority
                            )
                            # Note: Pour un vrai système, on traiterait le signal directement
                
                time.sleep(0.1)  # Petit délai pour éviter la surcharge CPU
                
            except Exception as e:
                print(f"❌ Erreur coordination: {e}")
                time.sleep(1.0)
    
    async def process_files_parallel(self, 
                                   file_paths: List[str], 
                                   priority: int = 1,
                                   callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Traite plusieurs fichiers en parallèle via les régions cérébrales"""
        
        task_id = f"task_{int(time.time()*1000)}_{hash(tuple(file_paths)) % 10000}"
        start_time = time.time()
        
        print(f"🧠 Démarrage traitement parallèle: {len(file_paths)} fichier(s)")
        
        # Pré-analyser les types de fichiers pour routage
        file_processor = UniversalFileProcessor()
        file_types = []
        for file_path in file_paths:
            file_type = file_processor.detector.detect_file_type(file_path)
            file_types.append((file_path, file_type))
        
        # Grouper par régions cérébrales appropriées
        region_tasks = defaultdict(list)
        for file_path, file_type in file_types:
            target_regions = self.task_router.get(file_type, [BrainRegion.EXECUTIVE_CORTEX])
            
            # Distribuer à la première région disponible
            for region_type in target_regions:
                if region_type in self.regions:
                    region_tasks[region_type].append(file_path)
                    break
        
        # Soumettre les tâches aux régions
        submitted_tasks = []
        for region_type, files in region_tasks.items():
            if files:
                task = ProcessingTask(
                    task_id=f"{task_id}_{region_type.value}",
                    region=region_type,
                    files=files,
                    priority=priority,
                    callback=callback
                )
                
                self.regions[region_type].submit_task(task)
                submitted_tasks.append((region_type, len(files)))
                
                print(f"  📤 {region_type.value}: {len(files)} fichier(s)")
        
        # Attendre la completion (avec timeout)
        timeout = 60.0  # 60 secondes max
        elapsed = 0.0
        check_interval = 0.5
        
        while elapsed < timeout:
            # Vérifier si toutes les queues sont vides
            all_done = True
            for region_type, _ in submitted_tasks:
                if self.regions[region_type].input_queue.qsize() > 0:
                    all_done = False
                    break
            
            if all_done:
                break
            
            await asyncio.sleep(check_interval)
            elapsed += check_interval
        
        # Collecter les résultats finaux
        all_output_signals = []
        for region in self.regions.values():
            signals = region.get_output_signals()
            all_output_signals.extend(signals)
        
        processing_time = time.time() - start_time
        self.total_tasks_processed += 1
        
        result = {
            'task_id': task_id,
            'processed_files': len(file_paths),
            'processing_time': processing_time,
            'output_signals': all_output_signals,
            'region_distribution': dict(region_tasks),
            'submitted_tasks': submitted_tasks,
            'status': 'completed' if elapsed < timeout else 'timeout'
        }
        
        print(f"✅ Traitement terminé: {processing_time:.2f}s ({result['status']})")
        
        return result
    
    def get_brain_status(self) -> Dict[str, Any]:
        """État complet du cerveau artificiel"""
        region_stats = {}
        for region_type, region in self.regions.items():
            region_stats[region_type.value] = region.get_stats()
        
        uptime = time.time() - self.start_time
        
        return {
            'uptime_seconds': uptime,
            'total_tasks_processed': self.total_tasks_processed,
            'regions': region_stats,
            'neural_network': self.neural_network.get_network_stats(),
            'is_active': self.is_active,
            'tasks_per_minute': (self.total_tasks_processed / max(1, uptime/60))
        }
    
    def shutdown(self):
        """Arrêt propre du cerveau"""
        print("🧠 Arrêt du cerveau artificiel...")
        self.is_active = False
        
        for region in self.regions.values():
            region.shutdown()
        
        print("✅ Arrêt terminé")

# Test de l'architecture
if __name__ == "__main__":
    async def test_brain_architecture():
        brain = BrainlikeParallelProcessor()
        
        # Test avec des fichiers de différents types
        test_files = [
            "README.md",  # Text
            # "test_image.jpg",  # Image  
            # "test_audio.wav",  # Audio
            # "test_data.csv",   # Data
            # "https://example.com"  # Web
        ]
        
        # Filtrer les fichiers qui existent
        existing_files = [f for f in test_files if os.path.exists(f) or f.startswith('http')]
        
        if existing_files:
            result = await brain.process_files_parallel(existing_files, priority=3)
            
            print("\n📊 Résultats:")
            print(f"  Fichiers traités: {result['processed_files']}")
            print(f"  Temps total: {result['processing_time']:.2f}s")
            print(f"  Signaux générés: {len(result['output_signals'])}")
            print(f"  Distribution: {result['region_distribution']}")
        
        # Afficher l'état du cerveau
        print("\n🧠 État du cerveau:")
        brain_status = brain.get_brain_status()
        print(f"  Temps de fonctionnement: {brain_status['uptime_seconds']:.1f}s")
        print(f"  Tâches traitées: {brain_status['total_tasks_processed']}")
        print(f"  Tâches/minute: {brain_status['tasks_per_minute']:.1f}")
        
        # Afficher les statistiques par région
        print("\n🎯 Régions cérébrales:")
        for region_name, stats in brain_status['regions'].items():
            print(f"  {region_name}:")
            print(f"    Traités: {stats['processed_count']}")
            print(f"    Erreurs: {stats['error_count']}")
            print(f"    Temps moyen: {stats['avg_processing_time']:.3f}s")
        
        brain.shutdown()
    
    import os
    asyncio.run(test_brain_architecture())