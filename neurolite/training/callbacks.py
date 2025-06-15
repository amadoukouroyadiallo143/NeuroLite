"""
Callbacks pour le Trainer NeuroLite.

Permet d'exécuter du code personnalisé à différentes étapes de l'entraînement.
"""

from typing import List, Dict, Any

class TrainerCallback:
    """Classe de base pour créer des callbacks."""
    
    def on_train_begin(self, state: Dict[str, Any]):
        pass

    def on_train_end(self, state: Dict[str, Any]):
        pass

    def on_epoch_begin(self, state: Dict[str, Any]):
        pass

    def on_epoch_end(self, state: Dict[str, Any]):
        pass

    def on_step_begin(self, state: Dict[str, Any]):
        pass

    def on_step_end(self, state: Dict[str, Any]):
        pass
        
    def on_evaluate_begin(self, state: Dict[str, Any]):
        pass
        
    def on_evaluate_end(self, state: Dict[str, Any], metrics: Dict[str, float]):
        pass

class CallbackHandler:
    """Gère une liste de callbacks et appelle leurs méthodes."""
    def __init__(self, callbacks: List[TrainerCallback]):
        self.callbacks = callbacks

    def on_train_begin(self, state: Dict[str, Any]):
        for callback in self.callbacks:
            callback.on_train_begin(state)

    def on_train_end(self, state: Dict[str, Any]):
        for callback in self.callbacks:
            callback.on_train_end(state)

    def on_epoch_begin(self, state: Dict[str, Any]):
        for callback in self.callbacks:
            callback.on_epoch_begin(state)

    def on_epoch_end(self, state: Dict[str, Any]):
        for callback in self.callbacks:
            callback.on_epoch_end(state)

    def on_step_begin(self, state: Dict[str, Any]):
        for callback in self.callbacks:
            callback.on_step_begin(state)

    def on_step_end(self, state: Dict[str, Any]):
        for callback in self.callbacks:
            callback.on_step_end(state)

    def on_evaluate_begin(self, state: Dict[str, Any]):
        for callback in self.callbacks:
            callback.on_evaluate_begin(state)

    def on_evaluate_end(self, state: Dict[str, Any], metrics: Dict[str, float]):
        for callback in self.callbacks:
            callback.on_evaluate_end(state, metrics)

class LoggingCallback(TrainerCallback):
    """Callback simple pour afficher les logs."""
    
    def on_step_end(self, state: Dict[str, Any]):
        # Affiche la perte toutes les N étapes
        if state['global_step'] % 100 == 0:
            print(f"Step {state['global_step']}: Loss = {state['loss']:.4f}")

    def on_epoch_end(self, state: Dict[str, Any]):
        print(f"Fin de l'Epoch {state['epoch']+1}")
        if 'eval_metrics' in state:
            print("Métriques d'évaluation:", state['eval_metrics']) 