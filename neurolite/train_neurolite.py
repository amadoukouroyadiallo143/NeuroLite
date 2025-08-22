#!/usr/bin/env python3
"""
NeuroLite AGI Training Script v2.0
Script principal pour l'entraînement de NeuroLite AGI avec toutes les phases.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import logging
import time

# Imports NeuroLite
from .training.trainer import NeuroLiteTrainer, create_trainer_callbacks, create_loggers
from .datasets import create_sample_dataloaders, create_specialized_dataloaders
from .Configs.config import create_tiny_config, create_development_config, NeuroLiteConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse les arguments de ligne de commande."""
    
    parser = argparse.ArgumentParser(description="NeuroLite AGI Training")
    
    # Configuration
    parser.add_argument("--config", type=str, choices=["tiny", "dev", "default"], 
                       default="tiny", help="Configuration à utiliser")
    parser.add_argument("--experiment-name", type=str, default=f"neurolite_exp_{int(time.time())}")
    
    # Entraînement
    parser.add_argument("--epochs", type=int, default=5, help="Nombre d'époques total")
    parser.add_argument("--batch-size", type=int, default=2, help="Taille de batch")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    
    # Données
    parser.add_argument("--dataset-type", type=str, choices=["multimodal", "text", "consciousness", "memory"],
                       default="multimodal", help="Type de dataset")
    parser.add_argument("--dataset-size", type=int, default=500, help="Taille du dataset")
    
    # GPU/Device
    parser.add_argument("--gpus", type=int, default=1 if torch.cuda.is_available() else 0)
    parser.add_argument("--accelerator", type=str, default="auto")
    
    # Phases d'entraînement
    parser.add_argument("--phase", type=str, choices=["foundation", "cognition", "higher_order", "integration", "end_to_end", "all"],
                       default="all", help="Phase d'entraînement spécifique")
    
    # Logging & Monitoring
    parser.add_argument("--use-wandb", action="store_true", help="Utiliser Wandb")
    parser.add_argument("--checkpoint-dir", type=str, default="./experiments/checkpoints")
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    
    # Mode de test
    parser.add_argument("--fast-dev-run", action="store_true", help="Test rapide (1 batch)")
    parser.add_argument("--dry-run", action="store_true", help="Test setup sans entraînement")
    
    return parser.parse_args()

def create_config(config_type: str) -> NeuroLiteConfig:
    """Crée la configuration selon le type."""
    
    if config_type == "tiny":
        return create_tiny_config()
    elif config_type == "dev":
        return create_development_config()
    else:
        return create_tiny_config()  # Default safe

def setup_training_environment(args):
    """Configure l'environnement d'entraînement."""
    
    # Créer dossiers
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path("./experiments/results").mkdir(parents=True, exist_ok=True)
    Path("./experiments/logs").mkdir(parents=True, exist_ok=True)
    
    # Configuration PyTorch
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info(f"🚀 GPU disponible: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("💻 Entraînement sur CPU")
    
    # Seed pour reproductibilité
    pl.seed_everything(42)

def create_trainer_instance(args, config: NeuroLiteConfig):
    """Crée l'instance du trainer."""
    
    # Callbacks
    callbacks = create_trainer_callbacks(args.experiment_name)
    
    # Loggers
    loggers = create_loggers(args.experiment_name, use_wandb=args.use_wandb)
    
    # Trainer PyTorch Lightning
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.gpus if args.gpus > 0 else "auto",
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=args.log_every_n_steps,
        enable_progress_bar=True,
        enable_model_summary=True,
        fast_dev_run=args.fast_dev_run,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        accumulate_grad_batches=4,  # Pour compenser les petits batches
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        default_root_dir=args.checkpoint_dir
    )
    
    return trainer

def main():
    """Fonction principale d'entraînement."""
    
    print("🚀 NEUROLITE AGI TRAINING v2.0")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    
    if args.dry_run:
        print("🧪 DRY RUN - Test configuration seulement")
    
    # Setup environnement
    setup_training_environment(args)
    
    # Configuration
    config = create_config(args.config)
    logger.info(f"📋 Configuration: {args.config}")
    logger.info(f"🧠 Modèle: {config.model_config.hidden_size}D, {config.model_config.num_layers} couches")
    
    # Dataset
    if args.dataset_type == "multimodal":
        train_loader, val_loader, _ = create_sample_dataloaders(
            batch_size=args.batch_size,
            num_workers=0  # 0 pour éviter problèmes multiprocessing
        )
    else:
        train_loader, val_loader = create_specialized_dataloaders(
            dataset_type=args.dataset_type,
            batch_size=args.batch_size
        )
    
    logger.info(f"📊 Dataset: {args.dataset_type}, Batches: Train={len(train_loader)}, Val={len(val_loader)}")
    
    if args.dry_run:
        # Test un batch
        try:
            sample_batch = next(iter(train_loader))
            logger.info(f"✅ Test batch réussi: {len(sample_batch)} éléments")
            print("✅ Configuration valide - arrêt dry run")
            return
        except Exception as e:
            logger.error(f"❌ Erreur test batch: {e}")
            return
    
    # Model
    model = NeuroLiteTrainer(
        config=config,
        use_wandb=args.use_wandb,
        experiment_name=args.experiment_name
    )
    
    # Trainer
    trainer = create_trainer_instance(args, config)
    
    # Informations d'entraînement
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 STATISTIQUES MODÈLE:")
    print(f"   • Paramètres totaux: {total_params:,}")
    print(f"   • Paramètres entraînables: {trainable_params:,}")
    print(f"   • Taille estimée: {total_params * 4 / 1024**2:.1f} MB")
    print(f"   • Phases d'entraînement: {len(model.training_phases)}")
    
    print(f"\n🎯 CONFIGURATION ENTRAÎNEMENT:")
    print(f"   • Époques: {args.epochs}")
    print(f"   • Batch size: {args.batch_size}")
    print(f"   • Learning rate: {args.learning_rate}")
    print(f"   • GPU: {'✅' if args.gpus > 0 else '❌'}")
    print(f"   • Wandb: {'✅' if args.use_wandb else '❌'}")
    
    # Démarrage entraînement
    print(f"\n🚀 DÉMARRAGE ENTRAÎNEMENT...")
    print("=" * 50)
    
    try:
        start_time = time.time()
        
        # Fit
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print("=" * 50)
        print("🎉 ENTRAÎNEMENT TERMINÉ!")
        print(f"⏱️  Temps total: {training_time:.1f}s ({training_time/60:.1f}min)")
        print(f"📁 Checkpoints: {args.checkpoint_dir}")
        
        # Test final optionnel
        if not args.fast_dev_run:
            print("\n🧪 Test final...")
            test_results = trainer.test(model, dataloaders=val_loader, verbose=True)
            print("✅ Test terminé")
        
    except KeyboardInterrupt:
        print("\n⚠️ Entraînement interrompu par l'utilisateur")
        
    except Exception as e:
        print(f"\n❌ Erreur durant l'entraînement: {e}")
        logger.error(f"Erreur entraînement: {e}", exc_info=True)
        
    finally:
        print(f"\n📊 Expérience: {args.experiment_name}")
        if args.use_wandb:
            print("🔗 Consultez Wandb pour les détails")

if __name__ == "__main__":
    main()