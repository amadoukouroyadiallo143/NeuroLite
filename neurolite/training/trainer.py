"""
NeuroLite AGI Trainer v2.0 - PyTorch Lightning
Trainer principal pour l'entraînement modulaire de NeuroLite AGI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
try:
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
except ImportError:
    # Fallback pour versions récentes
    from lightning.pytorch.loggers import TensorBoardLogger
    try:
        from lightning.pytorch.loggers import WandbLogger
    except ImportError:
        WandbLogger = None
import wandb
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

from ..core.agi_model import NeuroLiteAGI, AGIResponse, AGIMode
from ..Configs.config import NeuroLiteConfig, create_tiny_config
from .losses import AGIMultiModalLoss
from .metrics import AGIMetrics

logger = logging.getLogger(__name__)

@dataclass
class TrainingPhase:
    """Phase d'entraînement avec configuration spécifique."""
    name: str
    modules: List[str]  # Modules à entraîner
    epochs: int
    learning_rate: float
    weight_decay: float
    freeze_other_modules: bool = True
    
class NeuroLiteTrainer(pl.LightningModule):
    """
    Trainer PyTorch Lightning pour NeuroLite AGI.
    Support entraînement modulaire, multi-GPU, et métriques AGI.
    """
    
    def __init__(
        self,
        config: NeuroLiteConfig = None,
        training_phases: List[TrainingPhase] = None,
        use_wandb: bool = False,
        experiment_name: str = "neurolite_training"
    ):
        super().__init__()
        
        # Configuration
        self.config = config or create_tiny_config()
        self.training_phases = training_phases or self._create_default_phases()
        self.current_phase_idx = 0
        self.experiment_name = experiment_name
        
        # Modèle AGI
        self.agi_model = NeuroLiteAGI(config=self.config)
        
        # Systèmes d'entraînement
        self.loss_fn = AGIMultiModalLoss()
        self.metrics = AGIMetrics()
        
        # Tracking
        self.use_wandb = use_wandb
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
        # Métriques d'entraînement
        self.register_buffer("global_step_count", torch.tensor(0))
        self.register_buffer("epoch_count", torch.tensor(0))
        
        self.save_hyperparameters(ignore=['config'])
        
        logger.info(f"🚀 NeuroLiteTrainer initialisé avec {len(self.training_phases)} phases")
        
    def _create_default_phases(self) -> List[TrainingPhase]:
        """Crée les phases d'entraînement par défaut."""
        return [
            # Phase 1: Foundation (SSM + Multimodal)
            TrainingPhase(
                name="foundation", 
                modules=["ssm_core", "multimodal_fusion"],
                epochs=3,
                learning_rate=5e-4,
                weight_decay=1e-4
            ),
            
            # Phase 2: Cognition (Cognitive Core)
            TrainingPhase(
                name="cognition",
                modules=["cognitive_core"], 
                epochs=2,
                learning_rate=3e-4,
                weight_decay=1e-4
            ),
            
            # Phase 3: Higher-order (Consciousness + Memory)
            TrainingPhase(
                name="higher_order",
                modules=["consciousness", "memory"],
                epochs=2, 
                learning_rate=2e-4,
                weight_decay=1e-4
            ),
            
            # Phase 4: Integration (Reasoning + Interface)
            TrainingPhase(
                name="integration",
                modules=["reasoning", "unified_interface"],
                epochs=1,
                learning_rate=1e-4,
                weight_decay=1e-4
            ),
            
            # Phase 5: End-to-End Fine-tuning
            TrainingPhase(
                name="end_to_end",
                modules=["all"],
                epochs=3,
                learning_rate=1e-5,
                weight_decay=1e-5,
                freeze_other_modules=False
            )
        ]
    
    def get_current_phase(self) -> TrainingPhase:
        """Retourne la phase d'entraînement actuelle."""
        return self.training_phases[self.current_phase_idx]
    
    def advance_phase(self):
        """Passe à la phase suivante."""
        if self.current_phase_idx < len(self.training_phases) - 1:
            self.current_phase_idx += 1
            phase = self.get_current_phase()
            logger.info(f"📈 Passage à la phase: {phase.name}")
            self._freeze_modules(phase)
    
    def _freeze_modules(self, phase: TrainingPhase):
        """Gèle/dégèle les modules selon la phase."""
        if phase.modules[0] == "all":
            # Dégeler tout pour l'entraînement end-to-end
            for param in self.agi_model.parameters():
                param.requires_grad = True
            logger.info("🔓 Tous les modules dégelés pour end-to-end")
            return
            
        if phase.freeze_other_modules:
            # Geler tout d'abord
            for param in self.agi_model.parameters():
                param.requires_grad = False
            
            # Dégeler seulement les modules de cette phase
            for module_name in phase.modules:
                if hasattr(self.agi_model, module_name):
                    module = getattr(self.agi_model, module_name)
                    for param in module.parameters():
                        param.requires_grad = True
                        
            logger.info(f"🔒 Modules actifs: {phase.modules}")
    
    def configure_optimizers(self):
        """Configure optimiseur et scheduler selon la phase."""
        phase = self.get_current_phase()
        
        # Seulement les paramètres qui nécessitent des gradients
        trainable_params = [p for p in self.agi_model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=phase.learning_rate,
            weight_decay=phase.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=phase.epochs * 100,  # Warmup steps
            T_mult=2,
            eta_min=phase.learning_rate * 0.01
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
    
    def forward(self, batch: Dict[str, Any]) -> AGIResponse:
        """Forward pass du modèle."""
        task = batch.get("task", "training_forward")
        inputs = batch["inputs"]
        mode = batch.get("mode", AGIMode.LEARNING)
        
        return self.agi_model(task, inputs, mode=mode)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Étape d'entraînement."""
        phase = self.get_current_phase()
        
        # Forward pass
        response = self.forward(batch)
        
        # Calcul des pertes
        losses = self.loss_fn(response, batch["targets"])
        total_loss = losses["total_loss"]
        
        # Métriques
        metrics = self.metrics.compute_training_metrics(response, batch["targets"])
        
        # Logging
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/phase", float(self.current_phase_idx), on_step=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
        
        # Log losses individuelles
        for loss_name, loss_value in losses.items():
            if loss_name != "total_loss":
                self.log(f"train/{loss_name}", loss_value, on_step=True, on_epoch=True)
        
        # Log métriques AGI
        for metric_name, metric_value in metrics.items():
            self.log(f"train/{metric_name}", metric_value, on_step=True, on_epoch=True)
        
        # Stockage pour aggregation
        self.training_step_outputs.append({
            "loss": total_loss.detach(),
            "metrics": metrics,
            "phase": phase.name
        })
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Étape de validation."""
        
        # Forward pass (sans gradient)
        with torch.no_grad():
            response = self.forward(batch)
        
        # Calcul des pertes
        losses = self.loss_fn(response, batch["targets"])
        total_loss = losses["total_loss"]
        
        # Métriques
        metrics = self.metrics.compute_validation_metrics(response, batch["targets"])
        
        # Logging
        self.log("val/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log losses individuelles
        for loss_name, loss_value in losses.items():
            if loss_name != "total_loss":
                self.log(f"val/{loss_name}", loss_value, on_step=False, on_epoch=True)
        
        # Log métriques AGI
        for metric_name, metric_value in metrics.items():
            self.log(f"val/{metric_name}", metric_value, on_step=False, on_epoch=True)
        
        # Stockage pour aggregation
        self.validation_step_outputs.append({
            "loss": total_loss.detach(),
            "metrics": metrics
        })
        
        return total_loss
    
    def on_train_epoch_end(self):
        """Fin d'époque d'entraînement."""
        if len(self.training_step_outputs) == 0:
            return
            
        # Aggregation des métriques
        avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
        phase = self.get_current_phase()
        
        logger.info(f"📊 Époque {self.current_epoch} | Phase: {phase.name} | Loss: {avg_loss:.4f}")
        
        # Wandb logging spécial
        if self.use_wandb and wandb.run is not None:
            wandb.log({
                "epoch": self.current_epoch,
                "phase": phase.name,
                "avg_train_loss": avg_loss.item(),
                "trainable_params": sum(p.numel() for p in self.agi_model.parameters() if p.requires_grad)
            })
        
        # Clear outputs
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """Fin d'époque de validation."""
        if len(self.validation_step_outputs) == 0:
            return
            
        # Aggregation des métriques
        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        
        logger.info(f"📊 Validation | Loss: {avg_loss:.4f}")
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def on_train_end(self):
        """Fin de l'entraînement."""
        logger.info("🎉 Entraînement NeuroLite terminé!")
        
        if self.use_wandb and wandb.run is not None:
            wandb.log({"training_completed": True})
            
        # Sauvegarde finale
        self.save_final_checkpoint()
    
    def save_final_checkpoint(self):
        """Sauvegarde le checkpoint final."""
        save_path = f"./experiments/results/{self.experiment_name}_final.ckpt"
        
        checkpoint = {
            "model_state_dict": self.agi_model.state_dict(),
            "config": self.config,
            "phases_completed": self.current_phase_idx + 1,
            "total_phases": len(self.training_phases),
            "experiment_name": self.experiment_name
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"💾 Checkpoint final sauvegardé: {save_path}")


def create_trainer_callbacks(experiment_name: str) -> List[pl.Callback]:
    """Crée les callbacks pour l'entraînement."""
    
    callbacks = [
        # Checkpoint sur validation loss
        ModelCheckpoint(
            dirpath=f"./experiments/results/{experiment_name}/checkpoints",
            filename="neurolite-{epoch:02d}-{val/total_loss:.4f}",
            monitor="val/total_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            save_weights_only=False,
            verbose=True
        ),
        
        # Early stopping
        EarlyStopping(
            monitor="val/total_loss",
            min_delta=0.001,
            patience=5,
            verbose=True,
            mode="min"
        ),
        
        # Learning rate monitoring
        LearningRateMonitor(logging_interval="step"),
        
    ]
    
    return callbacks


def create_loggers(experiment_name: str, use_wandb: bool = False) -> List:
    """Crée les loggers pour l'entraînement."""
    
    loggers = [
        # TensorBoard (toujours)
        TensorBoardLogger(
            save_dir="./experiments/results",
            name=experiment_name,
            version=f"v_{int(time.time())}"
        )
    ]
    
    # Wandb optionnel
    if use_wandb:
        try:
            wandb_logger = WandbLogger(
                project="neurolite-agi",
                name=experiment_name,
                save_dir="./experiments/results"
            )
            loggers.append(wandb_logger)
            logger.info("🔗 Wandb logger activé")
        except Exception as e:
            logger.warning(f"⚠️ Impossible d'initialiser Wandb: {e}")
    
    return loggers