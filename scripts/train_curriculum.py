"""
Script principal pour l'entraînement avec Curriculum Learning du modèle NeuroLite.
"""
import torch
from pathlib import Path
from datasets import load_from_disk
import sys
import os
import argparse
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler

# Ajouter la racine du projet au sys.path pour les importations locales
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurolite.core.model import NeuroLiteModel
from neurolite.tokenization.tokenizer import NeuroLiteTokenizer
from neurolite.Configs.config import (
    NeuroLiteConfig, 
    ModelArchitectureConfig, 
    TrainingConfig, 
    TokenizerConfig, 
    MemoryConfig,
    ReasoningConfig,
    MMTextEncoderConfig,
    MMImageEncoderConfig,
    MMAudioEncoderConfig,
    MMVideoEncoderConfig,
    MMGraphEncoderConfig,
    MMTextDecoderConfig,
    MMImageDecoderConfig,
    MMAudioDecoderConfig,
    MMVideoDecoderConfig,
    MMGraphDecoderConfig
)
from neurolite.training.trainer import Trainer
from neurolite.training.utils import preprocess_sft_dataset
# --- NOUVEAUX IMPORTS POUR LE CURRICULUM LEARNING ---
from neurolite.training.difficulty_scorer import DifficultyScorer
from neurolite.training.curriculum_collator import CurriculumDataCollator

def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Entraînement avec Curriculum Learning pour NeuroLite")
    parser.add_argument("--dataset_path", type=str, default="data/raw/mixture_of_thoughts/train", help="Chemin vers le dataset.")
    parser.add_argument("--output_dir", type=str, default="./outputs/curriculum_learning", help="Répertoire de sortie.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Nombre d'époques d'entraînement.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Taille du batch d'entraînement par appareil.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Nombre de pas pour l'accumulation de gradient.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Taux d'apprentissage.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Pondération de la décroissance des poids.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Norme maximale pour le clipping de gradient.")
    parser.add_argument("--max_train_samples", type=int, default=0, help="Nombre max d'exemples (0 pour tous).")
    parser.add_argument("--save_steps", type=int, default=500, help="Fréquence de sauvegarde.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Fréquence de logging.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Type de planificateur.")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Nombre de pas pour le warmup.")
    # Arguments spécifiques au curriculum
    parser.add_argument("--scorer_alpha", type=float, default=1.0, help="Poids du score lexical dans le DifficultyScorer.")
    parser.add_argument("--scorer_beta", type=float, default=1.0, help="Poids du score de longueur dans le DifficultyScorer.")
    return parser.parse_args()

def count_parameters(model):
    """Compte et affiche le nombre de paramètres du modèle."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n--- Taille du Modèle ---")
    print(f"Paramètres totaux     : {total_params / 1e6:.2f} M")
    print(f"Paramètres entraînables : {trainable_params / 1e6:.2f} M")
    print("------------------------\n")
    return total_params, trainable_params

def main():
    args = parse_args()

    # --- 1. Configuration ---
    print("Configuration du modèle et de l'entraînement...")
    # ... (La configuration du modèle reste identique à celle de train_sft.py)
    model_config = ModelArchitectureConfig(
        hidden_size=512, num_hidden_layers=8,
        max_seq_length=1024,
        use_metacontroller=True, use_hierarchical_memory=True, use_continual_adapter=True,
        mm_text_encoder_config=MMTextEncoderConfig(),
        mm_image_encoder_config=MMImageEncoderConfig(),
        mm_audio_encoder_config=MMAudioEncoderConfig(),
        mm_video_encoder_config=MMVideoEncoderConfig(),
        mm_graph_encoder_config=MMGraphEncoderConfig(),
        mm_text_decoder_config=MMTextDecoderConfig(),
        mm_image_decoder_config=MMImageDecoderConfig(),
        mm_audio_decoder_config=MMAudioDecoderConfig(),
        mm_video_decoder_config=MMVideoDecoderConfig(),
        mm_graph_decoder_config=MMGraphDecoderConfig()
    )
    if model_config.mm_text_decoder_config:
        model_config.mm_text_decoder_config.max_seq_len = model_config.max_seq_length
        model_config.mm_text_decoder_config.input_dim = model_config.hidden_size
        model_config.mm_text_decoder_config.embedding_dim = model_config.hidden_size
        model_config.mm_text_decoder_config.num_heads = 8
        model_config.mm_text_decoder_config.hidden_dim = model_config.hidden_size * 4

    memory_config = MemoryConfig(use_external_memory=True, memory_dim=model_config.hidden_size)
    reasoning_config = ReasoningConfig(use_symbolic_module=True, use_causal_reasoning=True)
    training_config = TrainingConfig(
        output_dir=args.output_dir, num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm, save_steps=args.save_steps,
        logging_steps=args.logging_steps, lr_scheduler_type=args.lr_scheduler_type,
        num_warmup_steps=args.num_warmup_steps
    )
    tokenizer_config = TokenizerConfig(hidden_size=model_config.hidden_size)
    config = NeuroLiteConfig(
        model_config=model_config, training_config=training_config,
        tokenizer_config=tokenizer_config, memory_config=memory_config,
        reasoning_config=reasoning_config
    )

    # --- 2. Initialisation du Tokenizer ---
    print("Initialisation du NeuroLiteTokenizer...")
    tokenizer = NeuroLiteTokenizer(config.tokenizer_config, neurolite_config=config)
    
    # --- 3. Chargement des Données ---
    print(f"Chargement du dataset depuis {args.dataset_path}...")
    dataset = load_from_disk(Path(args.dataset_path))
    dataset = dataset.map(preprocess_sft_dataset, num_proc=4)

    if args.max_train_samples > 0:
        dataset = dataset.select(range(args.max_train_samples))
    
    split_dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset['train']
    # L'évaluation peut se faire sur le dataset complet
    eval_dataset = split_dataset['test']

    # On passe maintenant le dataset d'entraînement complet au collator
    # Il est conçu pour extraire les 'messages' et gérer la complexité.
    print(f"Dataset chargé: {len(train_dataset)} exemples pour l'entraînement.")

    # --- 4. Construction du Vocabulaire et Modèle ---
    # Le vocabulaire est construit sur le texte complet
    tokenizer.build_vocab(train_dataset, text_column='text', chat_format=True)
    config.model_config.mm_text_encoder_config.vocab_size = tokenizer.vocab_size
    if config.model_config.mm_text_decoder_config:
        config.model_config.mm_text_decoder_config.vocab_size = tokenizer.vocab_size

    print("Initialisation du modèle NeuroLite...")
    model = NeuroLiteModel(config, task_type='multimodal_generation', tokenizer=tokenizer)
    count_parameters(model)

    # --- 5. Configuration du Curriculum Learning ---
    print("Configuration du Curriculum Learning...")
    
    # a) Le Scorer de Difficulté
    difficulty_scorer = DifficultyScorer(alpha=args.scorer_alpha, beta=args.scorer_beta)

    # b) La Fonction de Progression (Pacing Function)
    # Augmente linéairement pour atteindre 100% à la fin du premier tiers de l'entraînement
    total_training_steps = (len(train_dataset) // args.per_device_train_batch_size) * args.num_train_epochs
    pacing_end_step = total_training_steps // 3 
    
    def linear_pacing(step: int) -> float:
        start_percent = 0.1  # Commencer avec les 10% de données les plus faciles
        if step >= pacing_end_step:
            return 1.0
        return start_percent + (1.0 - start_percent) * (step / pacing_end_step)

    # c) Le Data Collator pour le Curriculum
    # Il gère la tokenisation, le scoring et le tri en interne.
    curriculum_collator = CurriculumDataCollator(
        raw_dataset=list(train_dataset), # On passe la liste des exemples
        tokenizer=tokenizer,
        difficulty_scorer=difficulty_scorer,
        pacing_function=linear_pacing,
        max_seq_len=config.model_config.max_seq_length
    )

    # Le DataLoader D'ENTRAÎNEMENT NE DOIT PAS être mélangé.
    # Le CurriculumDataCollator gère le tirage aléatoire dans le pool de données autorisé.
    train_dataloader = DataLoader(
        train_dataset,  # Le contenu n'est pas utilisé directement, mais sa longueur si
        batch_size=config.training_config.per_device_train_batch_size,
        collate_fn=curriculum_collator,
        shuffle=False # TRÈS IMPORTANT pour le curriculum
    )

    # Le DataLoader d'évaluation peut rester un SFT collator classique (optionnel)
    # Pour la simplicité, on peut aussi le laisser utiliser le curriculum collator
    # qui, pour l'évaluation, pourrait utiliser 100% des données.
    # Ou nous pouvons créer un collator séparé pour l'évaluation.
    # Pour l'instant, nous nous concentrons sur l'entraînement.
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.training_config.per_device_train_batch_size,
        collate_fn=curriculum_collator # Réutiliser pour la simplicité
    )

    # --- 6. Optimiseur et Scheduler ---
    print("Configuration de l'optimiseur et du scheduler...")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=total_training_steps
    )

    model_forward_kwargs = {'update_memory': True, 'return_symbolic': True, 'continuous_learning': True}

    # --- 7. Entraînement ---
    print("Initialisation du Trainer...")
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        model_forward_kwargs=model_forward_kwargs
    )

    print("Début de l'entraînement avec Curriculum Learning...")
    trainer.train()

    print("Entraînement terminé.")
    print(f"Modèle sauvegardé dans : {config.training_config.output_dir}")

if __name__ == "__main__":
    main() 