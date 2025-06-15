"""
Script principal pour l'entraînement supervisé (SFT) du modèle NeuroLite
sur des datasets au format chat, comme Mixture of Thoughts.
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
from neurolite.training.data_collator import SFTDataCollator
from neurolite.training.utils import preprocess_sft_dataset

def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Entraînement SFT pour NeuroLite")
    parser.add_argument("--dataset_path", type=str, default="data/raw/mixture_of_thoughts/train", help="Chemin vers le dataset.")
    parser.add_argument("--output_dir", type=str, default="./outputs/sft_mixture_of_thoughts", help="Répertoire de sortie.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Nombre d'époques d'entraînement.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Taille du batch d'entraînement par appareil.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Nombre de pas pour l'accumulation de gradient.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Taux d'apprentissage.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Pondération de la décroissance des poids pour la régularisation.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Norme maximale pour le clipping de gradient.")
    parser.add_argument("--max_train_samples", type=int, default=0, help="Nombre max d'exemples d'entraînement (0 pour tous).")
    parser.add_argument("--save_steps", type=int, default=500, help="Fréquence de sauvegarde des checkpoints.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Fréquence de logging.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Type de planificateur (linear, cosine).")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Nombre de pas pour le warmup.")
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

    model_config = ModelArchitectureConfig(
        hidden_size=512, num_hidden_layers=8,
        max_seq_length=1024,
        # --- Activation de tous les modules ---
        use_metacontroller=True,
        use_hierarchical_memory=True,
        use_continual_adapter=True,
        # Encodeurs
        mm_text_encoder_config=MMTextEncoderConfig(),
        mm_image_encoder_config=MMImageEncoderConfig(),
        mm_audio_encoder_config=MMAudioEncoderConfig(),
        mm_video_encoder_config=MMVideoEncoderConfig(),
        mm_graph_encoder_config=MMGraphEncoderConfig(),
        # Décodeurs
        mm_text_decoder_config=MMTextDecoderConfig(),
        mm_image_decoder_config=MMImageDecoderConfig(),
        mm_audio_decoder_config=MMAudioDecoderConfig(),
        mm_video_decoder_config=MMVideoDecoderConfig(),
        mm_graph_decoder_config=MMGraphDecoderConfig()
    )

    # Assurer la cohérence des dimensions entre le modèle et les décodeurs
    if model_config.mm_text_decoder_config:
        model_config.mm_text_decoder_config.max_seq_len = model_config.max_seq_length
        # La dimension d'entrée du décodeur doit correspondre à la sortie du backbone
        model_config.mm_text_decoder_config.input_dim = model_config.hidden_size
        # La dimension interne du décodeur doit aussi correspondre pour éviter les conflits
        model_config.mm_text_decoder_config.embedding_dim = model_config.hidden_size
        # Assurer une configuration cohérente pour le décodeur Transformer
        model_config.mm_text_decoder_config.num_heads = 8  # Doit être un diviseur de hidden_size (512)
        model_config.mm_text_decoder_config.hidden_dim = model_config.hidden_size * 4 # Pratique standard (e.g., 512*4=2048)

    memory_config = MemoryConfig(
        use_external_memory=True,
        memory_dim=model_config.hidden_size
    )

    reasoning_config = ReasoningConfig(
        use_symbolic_module=True,
        use_causal_reasoning=True
    )

    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        num_warmup_steps=args.num_warmup_steps
    )

    # La configuration du tokenizer est maintenant centrale
    tokenizer_config = TokenizerConfig(hidden_size=model_config.hidden_size)

    config = NeuroLiteConfig(
        model_config=model_config,
        training_config=training_config,
        tokenizer_config=tokenizer_config,
        memory_config=memory_config,
        reasoning_config=reasoning_config
    )

    # --- 2. Initialisation du Tokenizer ---
    print("Initialisation du NeuroLiteTokenizer...")
    tokenizer = NeuroLiteTokenizer(config.tokenizer_config, neurolite_config=config)
    
    # --- 3. Chargement et Préparation des Données ---
    print(f"Chargement du dataset depuis {args.dataset_path}...")
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Le dataset n'a pas été trouvé à : {dataset_path}")
    dataset = load_from_disk(dataset_path)

    # --- ÉTAPE DE PRÉTRAITEMENT SUPPRIMÉE ---
    # Le SFTDataCollator gère maintenant directement le format 'messages'.
    # print("Prétraitement du dataset pour extraire prompt/response...")
    # dataset = dataset.map(preprocess_sft_dataset, num_proc=4)

    if args.max_train_samples > 0:
        print(f"Sélection d'un sous-ensemble de {args.max_train_samples} exemples pour le test...")
    
    split_dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    print(f"Dataset chargé: {len(train_dataset)} exemples d'entraînement, {len(eval_dataset)} exemples d'évaluation.")

    # --- ÉTAPE : Construction du vocabulaire du Tokenizer ---
    # Le tokenizer doit savoir qu'il traite un format de chat.
    tokenizer.build_vocab(train_dataset, text_column='text', chat_format=True)
    # Mettre à jour la configuration du modèle avec la nouvelle taille de vocabulaire
    config.model_config.mm_text_encoder_config.vocab_size = tokenizer.vocab_size
    if config.model_config.mm_text_decoder_config:
        config.model_config.mm_text_decoder_config.vocab_size = tokenizer.vocab_size

    # --- 4. Initialisation du Modèle et des DataLoaders ---
    print("Initialisation du modèle NeuroLite...")
    # Le tokenizer est passé directement au modèle
    # On active le mode 'multimodal_generation' pour inclure les décodeurs
    model = NeuroLiteModel(config, task_type='multimodal_generation', tokenizer=tokenizer)
    
    count_parameters(model)
    # Le data collator utilise maintenant notre tokenizer personnalisé
    data_collator = SFTDataCollator(
        tokenizer=tokenizer, 
        max_length=config.model_config.max_seq_length,
        label_ignore_index=-100 # Utiliser -100 pour ignorer les labels dans la loss
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training_config.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.training_config.per_device_train_batch_size,
        collate_fn=data_collator
    )

    # --- 5. Optimiseur et Scheduler ---
    print("Configuration de l'optimiseur et du scheduler...")
    optimizer = AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )

    num_training_steps = args.num_train_epochs * len(train_dataloader)
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Arguments supplémentaires pour la passe forward du modèle
    model_forward_kwargs = {
        'update_memory': True,
        'return_symbolic': True,
        'continuous_learning': True
    }

    # --- 6. Entraînement ---
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

    print("Début de l'entraînement...")
    trainer.train()

    print("Entraînement terminé.")
    print(f"Modèle sauvegardé dans : {config.training_config.output_dir}")

if __name__ == "__main__":
    main() 