"""
NeuroLite AGI v2.0 - Système de Monitoring Avancé
Surveillance complète des performances et interconnexions.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
import time
import threading
import queue
import logging
import json
import asyncio
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from enum import Enum
import psutil
import gc
from pathlib import Path

from .unified_interface import ModuleType, MessageType, Priority

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Niveaux d'alerte du système."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class PerformanceMetrics:
    """Métriques de performance complètes."""
    # Métriques temporelles
    avg_processing_time_ms: float
    min_processing_time_ms: float
    max_processing_time_ms: float
    processing_time_std_ms: float
    
    # Métriques de débit
    throughput_msg_per_sec: float
    throughput_tokens_per_sec: float
    
    # Métriques système
    cpu_usage_percent: float
    memory_usage_percent: float
    gpu_usage_percent: float
    gpu_memory_usage_mb: float
    
    # Métriques de qualité
    success_rate_percent: float
    error_rate_percent: float
    timeout_rate_percent: float
    
    # Métriques de coordination
    module_coordination_efficiency: float
    message_routing_efficiency: float
    cache_hit_rate_percent: float
    
    timestamp: float

@dataclass
class SystemAlert:
    """Alerte système."""
    level: AlertLevel
    component: str
    message: str
    timestamp: float
    metadata: Dict[str, Any]
    resolved: bool = False

class MetricsCollector:
    """Collecteur de métriques temps réel."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        
        # Historiques de métriques
        self.processing_times = deque(maxlen=window_size)
        self.throughput_history = deque(maxlen=window_size)
        self.error_counts = deque(maxlen=window_size)
        self.success_counts = deque(maxlen=window_size)
        
        # Métriques par module
        self.module_metrics = {module_type: {
            'processing_times': deque(maxlen=window_size),
            'success_count': 0,
            'error_count': 0,
            'last_activity': 0.0
        } for module_type in ModuleType}
        
        # Verrous pour thread-safety
        self.metrics_lock = threading.RLock()
        
        # Métriques système
        self.system_start_time = time.time()
        self.total_requests = 0
        
    def record_processing_time(self, module_type: ModuleType, 
                              processing_time_ms: float, success: bool = True):
        """Enregistre un temps de traitement."""
        with self.metrics_lock:
            self.processing_times.append(processing_time_ms)
            self.module_metrics[module_type]['processing_times'].append(processing_time_ms)
            self.module_metrics[module_type]['last_activity'] = time.time()
            
            if success:
                self.success_counts.append(1)
                self.module_metrics[module_type]['success_count'] += 1
            else:
                self.error_counts.append(1)
                self.module_metrics[module_type]['error_count'] += 1
            
            self.total_requests += 1
    
    def record_throughput(self, messages_per_sec: float, tokens_per_sec: float = 0.0):
        """Enregistre les métriques de débit."""
        with self.metrics_lock:
            self.throughput_history.append({
                'messages_per_sec': messages_per_sec,
                'tokens_per_sec': tokens_per_sec,
                'timestamp': time.time()
            })
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Calcule les métriques actuelles."""
        with self.metrics_lock:
            # Métriques temporelles
            if self.processing_times:
                processing_times_list = list(self.processing_times)
                avg_time = sum(processing_times_list) / len(processing_times_list)
                min_time = min(processing_times_list)
                max_time = max(processing_times_list)
                
                # Calcul de l'écart-type
                variance = sum((t - avg_time) ** 2 for t in processing_times_list) / len(processing_times_list)
                std_time = variance ** 0.5
            else:
                avg_time = min_time = max_time = std_time = 0.0
            
            # Métriques de débit
            current_throughput_msg = 0.0
            current_throughput_tokens = 0.0
            if self.throughput_history:
                recent_throughput = list(self.throughput_history)[-10:]  # 10 dernières mesures
                current_throughput_msg = sum(t['messages_per_sec'] for t in recent_throughput) / len(recent_throughput)
                current_throughput_tokens = sum(t['tokens_per_sec'] for t in recent_throughput) / len(recent_throughput)
            
            # Métriques système
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            gpu_percent = 0.0
            gpu_memory_mb = 0.0
            if torch.cuda.is_available():
                gpu_percent = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            
            # Métriques de qualité
            total_success = sum(self.success_counts)
            total_error = sum(self.error_counts)
            total_ops = total_success + total_error
            
            success_rate = (total_success / total_ops * 100) if total_ops > 0 else 100.0
            error_rate = (total_error / total_ops * 100) if total_ops > 0 else 0.0
            
            return PerformanceMetrics(
                avg_processing_time_ms=avg_time,
                min_processing_time_ms=min_time,
                max_processing_time_ms=max_time,
                processing_time_std_ms=std_time,
                throughput_msg_per_sec=current_throughput_msg,
                throughput_tokens_per_sec=current_throughput_tokens,
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory_percent,
                gpu_usage_percent=gpu_percent,
                gpu_memory_usage_mb=gpu_memory_mb,
                success_rate_percent=success_rate,
                error_rate_percent=error_rate,
                timeout_rate_percent=0.0,  # À implémenter
                module_coordination_efficiency=self._calculate_coordination_efficiency(),
                message_routing_efficiency=95.0,  # À implémenter
                cache_hit_rate_percent=85.0,  # À implémenter
                timestamp=time.time()
            )
    
    def _calculate_coordination_efficiency(self) -> float:
        """Calcule l'efficacité de coordination inter-modules."""
        active_modules = 0
        total_modules = len(self.module_metrics)
        
        current_time = time.time()
        
        for module_data in self.module_metrics.values():
            # Module considéré actif si activité dans les 10 dernières secondes
            if current_time - module_data['last_activity'] < 10.0:
                active_modules += 1
        
        return (active_modules / total_modules * 100) if total_modules > 0 else 0.0
    
    def get_module_health(self, module_type: ModuleType) -> Dict[str, Any]:
        """Retourne la santé d'un module spécifique."""
        with self.metrics_lock:
            module_data = self.module_metrics[module_type]
            
            total_ops = module_data['success_count'] + module_data['error_count']
            success_rate = (module_data['success_count'] / total_ops * 100) if total_ops > 0 else 100.0
            
            recent_times = list(module_data['processing_times'])[-100:]  # 100 dernières
            avg_time = sum(recent_times) / len(recent_times) if recent_times else 0.0
            
            # Statut de santé
            health_status = "healthy"
            if success_rate < 90:
                health_status = "degraded"
            if success_rate < 70:
                health_status = "critical"
            
            return {
                'health_status': health_status,
                'success_rate_percent': success_rate,
                'avg_processing_time_ms': avg_time,
                'total_operations': total_ops,
                'last_activity': module_data['last_activity'],
                'is_responsive': (time.time() - module_data['last_activity']) < 30.0
            }

class AlertManager:
    """Gestionnaire d'alertes système."""
    
    def __init__(self, max_alerts: int = 1000):
        self.max_alerts = max_alerts
        self.alerts = deque(maxlen=max_alerts)
        self.alert_counts = defaultdict(int)
        self.alert_lock = threading.RLock()
        
        # Seuils d'alerte
        self.thresholds = {
            'cpu_usage_warning': 80.0,
            'cpu_usage_critical': 95.0,
            'memory_usage_warning': 85.0,
            'memory_usage_critical': 95.0,
            'error_rate_warning': 5.0,
            'error_rate_critical': 15.0,
            'response_time_warning': 1000.0,  # ms
            'response_time_critical': 5000.0,  # ms
        }
    
    def check_thresholds(self, metrics: PerformanceMetrics):
        """Vérifie les seuils et génère des alertes."""
        current_time = time.time()
        
        # Vérification CPU
        if metrics.cpu_usage_percent > self.thresholds['cpu_usage_critical']:
            self.create_alert(AlertLevel.CRITICAL, "system", 
                            f"Utilisation CPU critique: {metrics.cpu_usage_percent:.1f}%",
                            {'cpu_usage': metrics.cpu_usage_percent})
        elif metrics.cpu_usage_percent > self.thresholds['cpu_usage_warning']:
            self.create_alert(AlertLevel.WARNING, "system",
                            f"Utilisation CPU élevée: {metrics.cpu_usage_percent:.1f}%",
                            {'cpu_usage': metrics.cpu_usage_percent})
        
        # Vérification mémoire
        if metrics.memory_usage_percent > self.thresholds['memory_usage_critical']:
            self.create_alert(AlertLevel.CRITICAL, "system",
                            f"Utilisation mémoire critique: {metrics.memory_usage_percent:.1f}%",
                            {'memory_usage': metrics.memory_usage_percent})
        elif metrics.memory_usage_percent > self.thresholds['memory_usage_warning']:
            self.create_alert(AlertLevel.WARNING, "system",
                            f"Utilisation mémoire élevée: {metrics.memory_usage_percent:.1f}%",
                            {'memory_usage': metrics.memory_usage_percent})
        
        # Vérification taux d'erreur
        if metrics.error_rate_percent > self.thresholds['error_rate_critical']:
            self.create_alert(AlertLevel.CRITICAL, "processing",
                            f"Taux d'erreur critique: {metrics.error_rate_percent:.1f}%",
                            {'error_rate': metrics.error_rate_percent})
        elif metrics.error_rate_percent > self.thresholds['error_rate_warning']:
            self.create_alert(AlertLevel.WARNING, "processing",
                            f"Taux d'erreur élevé: {metrics.error_rate_percent:.1f}%",
                            {'error_rate': metrics.error_rate_percent})
        
        # Vérification temps de réponse
        if metrics.avg_processing_time_ms > self.thresholds['response_time_critical']:
            self.create_alert(AlertLevel.CRITICAL, "performance",
                            f"Temps de réponse critique: {metrics.avg_processing_time_ms:.1f}ms",
                            {'response_time': metrics.avg_processing_time_ms})
        elif metrics.avg_processing_time_ms > self.thresholds['response_time_warning']:
            self.create_alert(AlertLevel.WARNING, "performance",
                            f"Temps de réponse élevé: {metrics.avg_processing_time_ms:.1f}ms",
                            {'response_time': metrics.avg_processing_time_ms})
    
    def create_alert(self, level: AlertLevel, component: str, 
                    message: str, metadata: Dict[str, Any]):
        """Crée une nouvelle alerte."""
        with self.alert_lock:
            alert = SystemAlert(
                level=level,
                component=component,
                message=message,
                timestamp=time.time(),
                metadata=metadata
            )
            
            self.alerts.append(alert)
            self.alert_counts[level] += 1
            
            # Log selon le niveau
            if level == AlertLevel.CRITICAL:
                logger.error(f"🚨 CRITIQUE - {component}: {message}")
            elif level == AlertLevel.WARNING:
                logger.warning(f"⚠️ ATTENTION - {component}: {message}")
            else:
                logger.info(f"ℹ️ INFO - {component}: {message}")
    
    def get_recent_alerts(self, max_count: int = 50) -> List[SystemAlert]:
        """Retourne les alertes récentes."""
        with self.alert_lock:
            return list(self.alerts)[-max_count:]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des alertes."""
        with self.alert_lock:
            recent_alerts = list(self.alerts)[-100:]  # 100 dernières
            
            return {
                'total_alerts': len(self.alerts),
                'recent_alerts_count': len(recent_alerts),
                'alert_counts_by_level': dict(self.alert_counts),
                'critical_alerts_last_hour': len([
                    a for a in recent_alerts 
                    if a.level == AlertLevel.CRITICAL and 
                       time.time() - a.timestamp < 3600
                ]),
                'unresolved_critical': len([
                    a for a in recent_alerts 
                    if a.level == AlertLevel.CRITICAL and not a.resolved
                ])
            }

class AdvancedMonitoringSystem:
    """Système de monitoring avancé complet."""
    
    def __init__(self, monitoring_interval: float = 1.0, 
                 enable_file_logging: bool = True,
                 log_directory: str = "./monitoring_logs"):
        
        self.monitoring_interval = monitoring_interval
        self.enable_file_logging = enable_file_logging
        self.log_directory = Path(log_directory)
        
        if enable_file_logging:
            self.log_directory.mkdir(exist_ok=True)
        
        # Composants principaux
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        
        # État du monitoring
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Historique des métriques pour analyse
        self.metrics_history = deque(maxlen=3600)  # 1 heure à 1s d'intervalle
        
        # Callbacks pour événements
        self.alert_callbacks = []
        self.metrics_callbacks = []
        
        logger.info("Système de monitoring avancé initialisé")
    
    def start_monitoring(self):
        """Démarre le monitoring en arrière-plan."""
        if self.is_monitoring:
            logger.warning("Monitoring déjà en cours")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            daemon=True, 
            name="AdvancedMonitoring"
        )
        self.monitoring_thread.start()
        
        logger.info("🔍 Monitoring avancé démarré")
    
    def stop_monitoring(self):
        """Arrête le monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Monitoring avancé arrêté")
    
    def _monitoring_loop(self):
        """Boucle principale de monitoring."""
        while self.is_monitoring:
            try:
                # Collecte des métriques
                current_metrics = self.metrics_collector.get_current_metrics()
                
                # Sauvegarde en historique
                self.metrics_history.append(current_metrics)
                
                # Vérification des seuils
                self.alert_manager.check_thresholds(current_metrics)
                
                # Callbacks métriques
                for callback in self.metrics_callbacks:
                    try:
                        callback(current_metrics)
                    except Exception as e:
                        logger.error(f"Erreur callback métriques: {e}")
                
                # Log vers fichier
                if self.enable_file_logging:
                    self._log_metrics_to_file(current_metrics)
                
                # Nettoyage mémoire périodique
                if len(self.metrics_history) % 300 == 0:  # Toutes les 5 minutes
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Erreur boucle monitoring: {e}")
                time.sleep(5.0)  # Attente plus longue en cas d'erreur
    
    def _log_metrics_to_file(self, metrics: PerformanceMetrics):
        """Enregistre les métriques dans un fichier."""
        try:
            log_file = self.log_directory / f"metrics_{time.strftime('%Y%m%d')}.jsonl"
            
            with open(log_file, 'a', encoding='utf-8') as f:
                metrics_dict = asdict(metrics)
                json.dump(metrics_dict, f)
                f.write('\n')
                
        except Exception as e:
            logger.error(f"Erreur logging fichier: {e}")
    
    def record_operation(self, module_type: ModuleType, 
                        processing_time_ms: float, success: bool = True):
        """Enregistre une opération pour monitoring."""
        self.metrics_collector.record_processing_time(module_type, processing_time_ms, success)
    
    def record_throughput(self, messages_per_sec: float, tokens_per_sec: float = 0.0):
        """Enregistre les métriques de débit."""
        self.metrics_collector.record_throughput(messages_per_sec, tokens_per_sec)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Retourne le statut complet du système."""
        current_metrics = self.metrics_collector.get_current_metrics()
        alert_summary = self.alert_manager.get_alert_summary()
        
        # Santé par module
        module_health = {}
        for module_type in ModuleType:
            module_health[module_type.value] = self.metrics_collector.get_module_health(module_type)
        
        # Statut global
        overall_status = "healthy"
        if alert_summary['unresolved_critical'] > 0:
            overall_status = "critical"
        elif alert_summary['critical_alerts_last_hour'] > 5:
            overall_status = "degraded"
        
        return {
            'overall_status': overall_status,
            'current_metrics': asdict(current_metrics),
            'alert_summary': alert_summary,
            'module_health': module_health,
            'monitoring_active': self.is_monitoring,
            'uptime_seconds': time.time() - self.metrics_collector.system_start_time
        }
    
    def get_performance_trends(self, hours: int = 1) -> Dict[str, Any]:
        """Analyse les tendances de performance."""
        if not self.metrics_history:
            return {'error': 'Pas de données historiques'}
        
        # Filtrer par période
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {'error': 'Pas de données pour la période demandée'}
        
        # Calculs de tendances
        response_times = [m.avg_processing_time_ms for m in recent_metrics]
        throughputs = [m.throughput_msg_per_sec for m in recent_metrics]
        error_rates = [m.error_rate_percent for m in recent_metrics]
        cpu_usage = [m.cpu_usage_percent for m in recent_metrics]
        
        return {
            'period_hours': hours,
            'data_points': len(recent_metrics),
            'response_time_trend': {
                'current': response_times[-1] if response_times else 0,
                'average': sum(response_times) / len(response_times) if response_times else 0,
                'min': min(response_times) if response_times else 0,
                'max': max(response_times) if response_times else 0,
            },
            'throughput_trend': {
                'current': throughputs[-1] if throughputs else 0,
                'average': sum(throughputs) / len(throughputs) if throughputs else 0,
                'peak': max(throughputs) if throughputs else 0,
            },
            'error_rate_trend': {
                'current': error_rates[-1] if error_rates else 0,
                'average': sum(error_rates) / len(error_rates) if error_rates else 0,
                'peak': max(error_rates) if error_rates else 0,
            },
            'resource_usage_trend': {
                'cpu_current': cpu_usage[-1] if cpu_usage else 0,
                'cpu_average': sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
                'cpu_peak': max(cpu_usage) if cpu_usage else 0,
            }
        }
    
    def add_alert_callback(self, callback: callable):
        """Ajoute un callback pour les alertes."""
        self.alert_callbacks.append(callback)
    
    def add_metrics_callback(self, callback: callable):
        """Ajoute un callback pour les métriques."""
        self.metrics_callbacks.append(callback)

# Export des classes principales
__all__ = [
    'AdvancedMonitoringSystem',
    'PerformanceMetrics',
    'SystemAlert',
    'AlertLevel',
    'MetricsCollector',
    'AlertManager'
]