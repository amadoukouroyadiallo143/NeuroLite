"""
Méta-contrôleur pour NeuroLite.

Ce module implémente un méta-contrôleur qui peut dynamiquement ajuster
la stratégie, l'architecture ou les hyperparamètres du modèle principal
en fonction de la tâche, des données ou de la performance.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from collections import deque
import random
import weakref

from neurolite.Configs.config import NeuroLiteConfig
from neurolite.continual.curriculum import CurriculumManager
# Forward declaration pour éviter l'import circulaire
if False:
    from neurolite.core.model import NeuroLiteModel


class MetaController(nn.Module):
    """
    Le MetaController observe l'état du modèle et de l'environnement
    pour prendre des décisions de haut niveau et adapter le modèle.
    """
    def __init__(self, model: 'NeuroLiteModel', config: NeuroLiteConfig, optimizer: Optional[torch.optim.Optimizer] = None):
        super().__init__()
        # Utiliser une référence faible pour éviter la récursion lors de l'initialisation de nn.Module
        self._model_ref = weakref.ref(model)
        self.config = config
        self.optimizer = optimizer
        
        # Exemple : un petit réseau pour décider de la stratégie
        # La dimension d'entrée pourrait inclure l'état du modèle + l'état de la tâche
        state_dim = config.model_config.hidden_size 
        self.strategy_selector = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Ex: 0=génération, 1=classification, 2=raisonnement
        )

    def forward(self, state_representation: torch.Tensor) -> Dict[str, Any]:
        """
        Prend une décision basée sur une représentation de l'état actuel.
        NOTE: Pour l'instant, cette méthode est simplifiée pour retourner une décision
        par défaut et éviter les problèmes de batch. La logique de sélection de stratégie
        sera développée ultérieurement.

        Args:
            state_representation: Un tenseur représentant l'état (actuellement ignoré).

        Returns:
            Un dictionnaire de décisions par défaut.
        """
        # Logique simplifiée pour les tests
        return {'task_type': 'base'}

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """Permet de lier l'optimiseur au contrôleur après l'initialisation."""
        self.optimizer = optimizer

    def set_learning_rate(self, new_lr: float):
        """Ajuste dynamiquement le learning rate de l'optimiseur."""
        if not self.optimizer:
            print("Avertissement: Aucun optimiseur n'est lié au MetaController.")
            return
        
        print(f"MetaController: ajustement du learning rate à {new_lr}")
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def set_dropout_rate(self, new_rate: float):
        """Ajuste le taux de dropout pour toutes les couches Dropout du modèle."""
        model = self._model_ref()
        if not model:
            print("Avertissement: La référence au modèle dans MetaController est perdue.")
            return

        print(f"MetaController: ajustement du taux de dropout à {new_rate}")
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = new_rate

    def toggle_modules_grad(self, modules_state: Dict[str, bool]):
        """
        Active ou désactive les gradients pour des modules spécifiques du modèle.

        Args:
            modules_state: Dictionnaire avec le nom du module (ex: 'symbolic_module', 'external_memory')
                           et un booléen (True pour activer grad, False pour désactiver).
        """
        model = self._model_ref()
        if not model:
            print("Avertissement: La référence au modèle dans MetaController est perdue.")
            return

        for module_name, requires_grad in modules_state.items():
            if hasattr(model, module_name):
                module = getattr(model, module_name)
                if module is not None:
                    print(f"MetaController: Réglage de requires_grad={requires_grad} pour {module_name}")
                    for param in module.parameters():
                        param.requires_grad = requires_grad
                else:
                    print(f"Avertissement: Module {module_name} non trouvé dans le modèle.")
            else:
                print(f"Avertissement: Attribut {module_name} non trouvé sur le modèle.")

    def adapt_on_performance(self, metrics: Dict[str, float]):
        """
        Adapte le modèle en fonction des métriques de performance.
        C'est une version plus avancée de l'ancienne méthode `adapt_model`.
        """
        model = self._model_ref()
        if not model:
            print("Avertissement: La référence au modèle dans MetaController est perdue.")
            return
            
        eval_loss = metrics.get('eval_loss')
        if eval_loss is not None:
            if eval_loss > 1.5: # Si la perte est très élevée
                self.set_learning_rate(self.config.training_config.learning_rate * 0.8) # Baisse agressive
                self.set_dropout_rate(min(0.5, model.config.model_config.dropout_rate + 0.1)) # Augmenter la régularisation
            elif eval_loss < 0.1: # Le modèle semble bien converger
                self.set_dropout_rate(max(0.0, model.config.model_config.dropout_rate - 0.05)) # Réduire la régularisation

    def select_modules(self, task_description: str) -> List[str]:
        """
        Sélectionne les modules pertinents (ex: mémoire, raisonnement) pour une tâche donnée.
        """
        # Logique de sélection de modules basée sur la description de la tâche (simplifiée)
        active_modules = ['core']
        if 'remember' in task_description or 'memory' in task_description:
            active_modules.append('memory')
        if 'reason' in task_description or 'cause' in task_description:
            active_modules.append('reasoning')
        if 'image' in task_description or 'audio' in task_description:
            active_modules.append('multimodal')
            
        return active_modules
