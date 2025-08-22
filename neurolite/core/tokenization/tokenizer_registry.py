"""
NeuroLite Tokenizer Registry System
===================================

Système de registre central pour gérer tous les tokenizers.
Permet l'enregistrement, la découverte et l'instanciation dynamiques.
"""

import time
import logging
from typing import Dict, List, Optional, Type, Any, Callable
from dataclasses import dataclass
from threading import Lock
import importlib
import inspect

from .base_tokenizer import BaseTokenizer, ModalityType, TokenizerConfig

logger = logging.getLogger(__name__)

@dataclass
class TokenizerRegistration:
    """Information d'enregistrement d'un tokenizer."""
    name: str
    modality: ModalityType
    tokenizer_class: Type[BaseTokenizer]
    priority: int = 0
    description: str = ""
    author: str = ""
    version: str = "1.0.0"
    requirements: List[str] = None
    factory_function: Optional[Callable] = None
    registration_time: float = 0.0
    
    def __post_init__(self):
        if self.requirements is None:
            self.requirements = []
        if self.registration_time == 0.0:
            self.registration_time = time.time()

class TokenizerRegistry:
    """Registre central pour tous les tokenizers."""
    
    def __init__(self):
        """Initialise le registre des tokenizers."""
        self._tokenizers: Dict[str, TokenizerRegistration] = {}
        self._by_modality: Dict[ModalityType, List[str]] = {}
        self._instances: Dict[str, BaseTokenizer] = {}
        self._lock = Lock()
        
        # Statistiques
        self.total_registrations = 0
        self.total_instantiations = 0
        self.failed_registrations = []
        
        logger.info("TokenizerRegistry initialisé")
    
    def register(self, 
                 name: str,
                 tokenizer_class: Type[BaseTokenizer],
                 modality: Optional[ModalityType] = None,
                 priority: int = 0,
                 description: str = "",
                 author: str = "",
                 version: str = "1.0.0",
                 requirements: Optional[List[str]] = None,
                 factory_function: Optional[Callable] = None,
                 overwrite: bool = False) -> bool:
        """
        Enregistre un nouveau tokenizer.
        
        Args:
            name: Nom unique du tokenizer
            tokenizer_class: Classe du tokenizer
            modality: Modalité (déduite si None)
            priority: Priorité (plus élevé = prioritaire)
            description: Description du tokenizer
            author: Auteur du tokenizer
            version: Version du tokenizer
            requirements: Dépendances requises
            factory_function: Fonction de création personnalisée
            overwrite: Autoriser l'écrasement
            
        Returns:
            bool: True si enregistrement réussi
        """
        with self._lock:
            try:
                # Vérifier si déjà enregistré
                if name in self._tokenizers and not overwrite:
                    logger.warning(f"Tokenizer '{name}' déjà enregistré")
                    return False
                
                # Valider la classe
                if not self._validate_tokenizer_class(tokenizer_class):
                    return False
                
                # Déduire la modalité si non fournie
                if modality is None:
                    modality = self._deduce_modality(tokenizer_class)
                
                # Créer l'enregistrement
                registration = TokenizerRegistration(
                    name=name,
                    modality=modality,
                    tokenizer_class=tokenizer_class,
                    priority=priority,
                    description=description,
                    author=author,
                    version=version,
                    requirements=requirements or [],
                    factory_function=factory_function
                )
                
                # Enregistrer
                self._tokenizers[name] = registration
                
                # Indexer par modalité
                if modality not in self._by_modality:
                    self._by_modality[modality] = []
                if name not in self._by_modality[modality]:
                    self._by_modality[modality].append(name)
                
                # Trier par priorité
                self._by_modality[modality].sort(
                    key=lambda n: self._tokenizers[n].priority, 
                    reverse=True
                )
                
                self.total_registrations += 1
                logger.info(f"Tokenizer '{name}' enregistré pour modalité {modality.value}")
                return True
                
            except Exception as e:
                error_msg = f"Erreur enregistrement tokenizer '{name}': {e}"
                logger.error(error_msg)
                self.failed_registrations.append({
                    'name': name,
                    'error': str(e),
                    'timestamp': time.time()
                })
                return False
    
    def get_tokenizer(self, 
                     name: str, 
                     config: Optional[TokenizerConfig] = None,
                     cache_instance: bool = True) -> Optional[BaseTokenizer]:
        """
        Récupère ou crée une instance de tokenizer.
        
        Args:
            name: Nom du tokenizer
            config: Configuration
            cache_instance: Mettre en cache l'instance
            
        Returns:
            BaseTokenizer ou None si non trouvé
        """
        with self._lock:
            try:
                # Vérifier si enregistré
                if name not in self._tokenizers:
                    logger.error(f"Tokenizer '{name}' non trouvé")
                    return None
                
                # Retourner instance cachée si disponible
                if cache_instance and name in self._instances:
                    return self._instances[name]
                
                # Créer nouvelle instance
                registration = self._tokenizers[name]
                
                # Vérifier les dépendances
                if not self._check_requirements(registration.requirements):
                    logger.error(f"Dépendances manquantes pour '{name}'")
                    return None
                
                # Créer l'instance
                if registration.factory_function:
                    instance = registration.factory_function(config)
                else:
                    if config is None:
                        config = TokenizerConfig()
                    instance = registration.tokenizer_class(config, registration.modality)
                
                # Mettre en cache si demandé
                if cache_instance:
                    self._instances[name] = instance
                
                self.total_instantiations += 1
                logger.debug(f"Instance tokenizer '{name}' créée")
                return instance
                
            except Exception as e:
                logger.error(f"Erreur création instance '{name}': {e}")
                return None
    
    def get_tokenizer_for_modality(self, 
                                  modality: ModalityType,
                                  config: Optional[TokenizerConfig] = None,
                                  strategy: Optional[str] = None) -> Optional[BaseTokenizer]:
        """
        Récupère le meilleur tokenizer pour une modalité.
        
        Args:
            modality: Type de modalité
            config: Configuration
            strategy: Stratégie préférée (optionnel)
            
        Returns:
            BaseTokenizer ou None
        """
        with self._lock:
            if modality not in self._by_modality:
                logger.error(f"Aucun tokenizer pour modalité {modality.value}")
                return None
            
            # Liste des tokenizers pour cette modalité (déjà triés par priorité)
            tokenizer_names = self._by_modality[modality]
            
            # Si stratégie spécifiée, chercher correspondance
            if strategy:
                for name in tokenizer_names:
                    registration = self._tokenizers[name]
                    tokenizer = self.get_tokenizer(name, config)
                    if tokenizer and strategy in [s.value for s in tokenizer.get_supported_strategies()]:
                        return tokenizer
            
            # Sinon, retourner le premier (plus haute priorité)
            return self.get_tokenizer(tokenizer_names[0], config)
    
    def list_tokenizers(self, modality: Optional[ModalityType] = None) -> List[Dict[str, Any]]:
        """
        Liste tous les tokenizers enregistrés.
        
        Args:
            modality: Filtrer par modalité (optionnel)
            
        Returns:
            Liste des informations de tokenizers
        """
        with self._lock:
            result = []
            
            for name, registration in self._tokenizers.items():
                if modality is None or registration.modality == modality:
                    result.append({
                        'name': name,
                        'modality': registration.modality.value,
                        'class': registration.tokenizer_class.__name__,
                        'priority': registration.priority,
                        'description': registration.description,
                        'author': registration.author,
                        'version': registration.version,
                        'requirements': registration.requirements,
                        'has_factory': registration.factory_function is not None,
                        'is_cached': name in self._instances,
                        'registration_time': registration.registration_time
                    })
            
            # Trier par modalité puis priorité
            result.sort(key=lambda x: (x['modality'], -x['priority']))
            return result
    
    def unregister(self, name: str) -> bool:
        """
        Désenregistre un tokenizer.
        
        Args:
            name: Nom du tokenizer
            
        Returns:
            bool: True si succès
        """
        with self._lock:
            if name not in self._tokenizers:
                return False
            
            registration = self._tokenizers[name]
            
            # Retirer du registre principal
            del self._tokenizers[name]
            
            # Retirer de l'index par modalité
            if registration.modality in self._by_modality:
                if name in self._by_modality[registration.modality]:
                    self._by_modality[registration.modality].remove(name)
                
                # Nettoyer la liste si vide
                if not self._by_modality[registration.modality]:
                    del self._by_modality[registration.modality]
            
            # Retirer de l'instance cache
            if name in self._instances:
                del self._instances[name]
            
            logger.info(f"Tokenizer '{name}' désenregistré")
            return True
    
    def clear_cache(self):
        """Vide le cache des instances."""
        with self._lock:
            self._instances.clear()
            logger.info("Cache des instances vidé")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques du registre."""
        with self._lock:
            by_modality_count = {
                modality.value: len(names) 
                for modality, names in self._by_modality.items()
            }
            
            return {
                'total_registered': len(self._tokenizers),
                'total_instantiations': self.total_instantiations,
                'cached_instances': len(self._instances),
                'by_modality': by_modality_count,
                'failed_registrations': len(self.failed_registrations),
                'recent_failures': self.failed_registrations[-5:] if self.failed_registrations else []
            }
    
    def _validate_tokenizer_class(self, tokenizer_class: Type[BaseTokenizer]) -> bool:
        """Valide qu'une classe est un tokenizer valide."""
        try:
            # Vérifier que c'est une sous-classe de BaseTokenizer
            if not issubclass(tokenizer_class, BaseTokenizer):
                logger.error(f"Classe {tokenizer_class.__name__} n'hérite pas de BaseTokenizer")
                return False
            
            # Vérifier les méthodes requises
            required_methods = ['tokenize', 'detokenize', 'get_vocab_size', 'get_supported_strategies']
            for method in required_methods:
                if not hasattr(tokenizer_class, method):
                    logger.error(f"Méthode '{method}' manquante dans {tokenizer_class.__name__}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur validation classe: {e}")
            return False
    
    def _deduce_modality(self, tokenizer_class: Type[BaseTokenizer]) -> ModalityType:
        """Déduit la modalité à partir du nom de la classe."""
        class_name = tokenizer_class.__name__.lower()
        
        if 'text' in class_name:
            return ModalityType.TEXT
        elif 'image' in class_name:
            return ModalityType.IMAGE
        elif 'audio' in class_name:
            return ModalityType.AUDIO
        elif 'video' in class_name:
            return ModalityType.VIDEO
        elif 'structured' in class_name:
            return ModalityType.STRUCTURED
        else:
            return ModalityType.UNKNOWN
    
    def _check_requirements(self, requirements: List[str]) -> bool:
        """Vérifie que toutes les dépendances sont disponibles."""
        for requirement in requirements:
            try:
                importlib.import_module(requirement)
            except ImportError:
                logger.error(f"Dépendance manquante: {requirement}")
                return False
        return True
    
    def auto_discover_tokenizers(self, package_paths: List[str] = None) -> int:
        """
        Découvre et enregistre automatiquement les tokenizers.
        
        Args:
            package_paths: Chemins de packages à explorer
            
        Returns:
            int: Nombre de tokenizers découverts
        """
        if package_paths is None:
            package_paths = ['neurolite.core.tokenization.tokenizers']
        
        discovered = 0
        
        for package_path in package_paths:
            try:
                package = importlib.import_module(package_path)
                
                for name in dir(package):
                    obj = getattr(package, name)
                    
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseTokenizer) and 
                        obj != BaseTokenizer):
                        
                        # Auto-enregistrement
                        tokenizer_name = name.lower().replace('tokenizer', '')
                        if self.register(tokenizer_name, obj):
                            discovered += 1
                            
            except Exception as e:
                logger.warning(f"Erreur découverte automatique dans {package_path}: {e}")
        
        logger.info(f"{discovered} tokenizers découverts automatiquement")
        return discovered

# Instance globale du registre
_global_registry = TokenizerRegistry()

def register_tokenizer(name: str, tokenizer_class: Type[BaseTokenizer], **kwargs) -> bool:
    """
    Fonction utilitaire pour enregistrer un tokenizer dans le registre global.
    
    Args:
        name: Nom du tokenizer
        tokenizer_class: Classe du tokenizer
        **kwargs: Arguments additionnels
        
    Returns:
        bool: True si succès
    """
    return _global_registry.register(name, tokenizer_class, **kwargs)

def get_global_registry() -> TokenizerRegistry:
    """Retourne l'instance du registre global."""
    return _global_registry