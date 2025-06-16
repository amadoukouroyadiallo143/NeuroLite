"""
Classe Trainer pour gérer l'entraînement et l'évaluation des modèles NeuroLite.
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import os
from typing import List, Dict, Optional, Any
import wandb
import time

from neurolite.core.model import NeuroLiteModel
from neurolite.Configs.config import NeuroLiteConfig
from neurolite.meta.metacontroller import MetaController
from neurolite.continual.curriculum import CurriculumManager
from .metrics import compute_generative_metrics
from .callbacks import TrainerCallback, CallbackHandler

class Trainer:
    def __init__(
        self,
        model: NeuroLiteModel,
        config: NeuroLiteConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[LambdaLR] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        metacontroller: Optional[MetaController] = None,
        curriculum_manager: Optional[CurriculumManager] = None,
        model_forward_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_writer: Optional[Any] = None
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = config.device
        self.model.to(self.device)

        self.optimizer = optimizer or AdamW(self.model.parameters(), lr=config.training_config.learning_rate)
        self.scheduler = scheduler # Le scheduler peut être complexe, laisser l'utilisateur le créer
        
        self.callback_handler = CallbackHandler(callbacks or [])
        self.metacontroller = metacontroller
        if self.metacontroller:
            self.metacontroller.set_optimizer(self.optimizer) # Lier l'optimiseur

        self.curriculum_manager = curriculum_manager
        self.global_step = 0
        self.state = {} # Pour stocker l'état de l'entraînement
        self.model_forward_kwargs = model_forward_kwargs or {}
        self.tensorboard_writer = tensorboard_writer

    def _prepare_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Déplace un batch de données sur le bon appareil."""
        prepared_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                prepared_batch[k] = v.to(self.device)
            elif isinstance(v, dict):
                # Gérer les dictionnaires imbriqués (ex: pour les entrées multimodales)
                prepared_batch[k] = self._prepare_inputs(v)
            else:
                prepared_batch[k] = v
        return prepared_batch

    def _find_tensor_in_batch(self, batch_data: Any) -> Optional[torch.Tensor]:
        """Trouve de manière récursive le premier tenseur dans une structure de données de batch."""
        if isinstance(batch_data, torch.Tensor):
            return batch_data
        if isinstance(batch_data, dict):
            for value in batch_data.values():
                tensor = self._find_tensor_in_batch(value)
                if tensor is not None:
                    return tensor
        elif isinstance(batch_data, (list, tuple)):
            for item in batch_data:
                tensor = self._find_tensor_in_batch(item)
                if tensor is not None:
                    return tensor
        return None

    def train(self):
        """Lance la boucle d'entraînement principale."""
        self.callback_handler.on_train_begin(self.state)

        for epoch in range(self.config.training_config.num_train_epochs):
            self.state['epoch'] = epoch
            self.callback_handler.on_epoch_begin(self.state)
            
            self.model.train()
            
            # Créer une barre de progression TQDM pour suivre l'époque
            progress_bar = tqdm(self.train_dataloader, desc=f"Époque {epoch + 1}/{self.config.training_config.num_train_epochs}")
            
            for step, batch in enumerate(progress_bar):
                self.state['step'] = step
                self.callback_handler.on_step_begin(self.state)
                
                t0 = time.time()
                inputs = self._prepare_inputs(batch)
                t1 = time.time()
                
                outputs = self.model(**inputs, **self.model_forward_kwargs)
                t2 = time.time()

                loss = outputs.get('loss') # Utiliser .get() pour la sécurité
                
                if loss is None:
                    print(f"Avertissement: La perte est nulle à l'étape {step}. Le batch sera ignoré.")
                    continue

                loss.backward()
                t3 = time.time()
                
                if self.global_step == 0:
                    print(f"\n--- Diagnostic de Temps (Première Étape) ---")
                    print(f"Préparation des entrées : {t1 - t0:.4f}s")
                    print(f"Passe forward du modèle: {t2 - t1:.4f}s")
                    print(f"Passe backward (loss.backward()): {t3 - t2:.4f}s")
                    print(f"---------------------------------------------\n")

                if (step + 1) % self.config.training_config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training_config.max_grad_norm)
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                
                self.global_step += 1
                self.state['loss'] = loss.item()
                
                # Mettre à jour la barre de progression avec la perte actuelle
                progress_bar.set_postfix({'loss': loss.item()})
                
                # --- LOGGING TENSORBOARD ---
                if self.tensorboard_writer and self.global_step % self.config.training_config.logging_steps == 0:
                    self.tensorboard_writer.add_scalar('Loss/train', loss.item(), self.global_step)
                    if self.scheduler:
                        self.tensorboard_writer.add_scalar('LearningRate', self.scheduler.get_last_lr()[0], self.global_step)
                # -------------------------

                self.callback_handler.on_step_end(self.state)
            
            # Évaluation à la fin de chaque époque
            if self.eval_dataloader:
                eval_metrics = self.evaluate()
                self.state['eval_metrics'] = eval_metrics
                
                # Afficher les métriques d'évaluation de manière claire
                print(f"\n--- Résumé de l'Évaluation (Époque {epoch + 1}) ---")
                for metric, value in eval_metrics.items():
                    # Formatter le nom de la métrique pour une meilleure lisibilité
                    metric_name = metric.replace('_', ' ').capitalize()
                    print(f"  {metric_name:<20}: {value:.4f}")
                    # --- LOGGING TENSORBOARD ---
                    if self.tensorboard_writer:
                        self.tensorboard_writer.add_scalar(f'Evaluation/{metric_name}', value, self.global_step)
                    # -------------------------
                print("------------------------------------------\n")
                
                # Mettre à jour les modules dynamiques
                if self.curriculum_manager:
                    self.curriculum_manager.update_competence(eval_metrics)
                if self.metacontroller:
                    self.metacontroller.adapt_on_performance(eval_metrics)

            self.callback_handler.on_epoch_end(self.state)

        self.callback_handler.on_train_end(self.state)

        # Sauvegarder le modèle final
        save_directory = self.config.training_config.output_dir
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        print(f"\nSauvegarde du modèle final dans {save_directory}...")
        self.model.save_pretrained(save_directory)
        print("Modèle sauvegardé avec succès.")

    def evaluate(self) -> Dict[str, float]:
        """Lance la boucle d'évaluation."""
        self.callback_handler.on_evaluate_begin(self.state)
        
        self.model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                inputs = self._prepare_inputs(batch)
                outputs = self.model(**inputs, **self.model_forward_kwargs)
                
                loss = outputs['loss']
                if loss is not None:
                    # Trouver un tenseur dans le batch pour déterminer la taille du batch
                    ref_tensor = self._find_tensor_in_batch(inputs)
                    if ref_tensor is None:
                        raise ValueError("Impossible de déterminer la taille du batch car aucun tenseur n'a été trouvé dans les entrées.")
                    
                    batch_size = ref_tensor.size(0)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size

        # Calculer la perte moyenne
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        
        metrics = compute_generative_metrics(avg_loss)
        
        self.callback_handler.on_evaluate_end(self.state, metrics)
        return metrics 