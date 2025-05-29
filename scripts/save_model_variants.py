#!/usr/bin/env python3
"""
Script pour générer et sauvegarder différentes variantes de modèles NeuroLite.

Ce script crée plusieurs tailles de modèles adaptées à différents cas d'utilisation,
puis les sauvegarde dans des dossiers distincts pour une publication sur Hugging Face.

Les configurations sont basées sur l'architecture NeuroLite et suivent les bonnes pratiques
du projet, en utilisant les configurations prédéfinies et en assurant la cohérence
des paramètres entre les différentes composantes du modèle.
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

# Ajouter le répertoire parent au chemin
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from neurolite.core.model import NeuroLiteModel
from neurolite.Configs.config import (
    NeuroLiteConfig, 
    ModelArchitectureConfig, 
    TrainingConfig,
    TokenizerConfig,
    MemoryConfig,
    LoggingConfig,
    TextEncoderConfig,
    ImageEncoderConfig,
    AudioEncoderConfig,
    VideoEncoderConfig,
    GraphEncoderConfig,
    QuantizerConfig,
    TextEncoderConfig,
    ImageEncoderConfig,
    AudioEncoderConfig,
    VideoEncoderConfig,
    GraphEncoderConfig
)

def get_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Calcule et retourne la taille du modèle en octets, Mo et Go.
    
    Args:
        model: Modèle PyTorch à analyser
        
    Returns:
        Dictionnaire contenant la taille en octets, Mo et Go, ainsi que le nombre de paramètres
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    return {
        'bytes': total_size,
        'mb': total_size / (1024 ** 2),
        'gb': total_size / (1024 ** 3),
        'num_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

def create_model_config(variant: str) -> NeuroLiteConfig:
    """
    Crée une configuration de modèle selon la variante spécifiée.
    
    Args:
        variant: Variante du modèle ('tiny', 'small', 'base', 'large')
        
    Returns:
        Instance de NeuroLiteConfig configurée pour la variante demandée
        
    Raises:
        ValueError: Si la variante spécifiée n'est pas reconnue
    """
    # Configurations de base pour chaque variante
    configs = {
        'tiny': {
            'hidden_size': 128,
            'num_hidden_layers': 4,
            'num_attention_heads': 4,
            'intermediate_size': 512,
            'memory_dim': 128,
            'memory_size': 32,
            'num_memory_heads': 2,
            'use_external_memory': False,
            'use_dynamic_routing': False,
            'use_multimodal_input': False,
            'use_continual_adapter': False,
            'max_position_embeddings': 2048,
            'batch_size': 8,
            'learning_rate': 5e-5,
            'warmup_steps': 1000,
        },
        'small': {
            'hidden_size': 384,
            'num_hidden_layers': 6,
            'num_attention_heads': 6,
            'intermediate_size': 1536,
            'memory_dim': 256,
            'memory_size': 64,
            'num_memory_heads': 4,
            'use_external_memory': True,
            'use_dynamic_routing': False,
            'use_multimodal_input': False,
            'use_continual_adapter': False,
            'max_position_embeddings': 4096,
            'batch_size': 8,
            'learning_rate': 3e-5,
            'warmup_steps': 2000,
        },
        'base': {
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'memory_dim': 512,
            'memory_size': 128,
            'num_memory_heads': 8,
            'use_external_memory': True,
            'use_dynamic_routing': True,
            'use_multimodal_input': True,
            'use_continual_adapter': True,
            'max_position_embeddings': 8192,
            'batch_size': 4,
            'learning_rate': 2e-5,
            'warmup_steps': 4000,
        },
        'large': {
            'hidden_size': 1024,
            'num_hidden_layers': 24,
            'num_attention_heads': 16,
            'intermediate_size': 4096,
            'memory_dim': 1024,
            'memory_size': 256,
            'num_memory_heads': 16,
            'use_external_memory': True,
            'use_dynamic_routing': True,
            'use_multimodal_input': True,
            'use_continual_adapter': True,
            'max_position_embeddings': 16384,
            'batch_size': 2,
            'learning_rate': 1e-5,
            'warmup_steps': 10000,
        }
    }
    
    if variant not in configs:
        raise ValueError(f"Variante inconnue: {variant}. Choisissez parmi: {list(configs.keys())}")
    
    config = configs[variant]
    
    # Configuration du tokenizer
    tokenizer_config = TokenizerConfig(
        vocab_size=50000,
        max_length=config.get('max_position_embeddings', 512),
        padding_side='right',
        truncation=True
    )
    
    # Configuration de la mémoire
    memory_config = MemoryConfig(
        use_external_memory=config.get('use_external_memory', True),
        memory_size=config.get('memory_size', 64),
        memory_dim=config.get('memory_dim', 256),
        num_memory_heads=config.get('num_memory_heads', 4),
        memory_update_mechanism=config.get('memory_update_mechanism', 'gru')
    )
    
    # Configuration de l'architecture du modèle
    model_config = ModelArchitectureConfig(
        # Paramètres principaux
        hidden_size=config.get('hidden_size', 768),
        num_hidden_layers=config.get('num_hidden_layers', 12),
        num_attention_heads=config.get('num_attention_heads', 12),
        intermediate_size=config.get('intermediate_size', 3072),
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=config.get('max_position_embeddings', 512),
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        max_seq_length=config.get('max_position_embeddings', 512),
        
        # Configuration de l'encodeur d'entrée
        input_projection_type="minhash_bloom",
        minhash_num_permutations=128,
        bloom_filter_size=512,
        vocab_size=50000,
        
        # Configuration MLP-Mixer
        num_mixer_layers=6,
        token_mixing_hidden_size=512,
        channel_mixing_hidden_size=1024,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        activation="gelu",
        
        # Configuration mémoire externe
        use_external_memory=config.get('use_external_memory', True)
    )
    
    # Configuration d'entraînement
    training_config = TrainingConfig(
        # Chemins des données
        output_dir=str(PROJECT_ROOT / "models"),
        train_data_path="data/processed/train",
        val_data_path="data/processed/val",
        logging_dir=str(PROJECT_ROOT / "logs"),
        
        # Hyperparamètres d'entraînement
        learning_rate=config.get('learning_rate', 5e-5),
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Configuration du batch size
        per_device_train_batch_size=config.get('batch_size', 8),
        per_device_eval_batch_size=config.get('batch_size', 8) * 2,
        gradient_accumulation_steps=1,
        
        # Configuration de l'apprentissage
        num_train_epochs=10,
        warmup_steps=config.get('warmup_steps', 1000),
        
        # Configuration de la précision
        fp16=torch.cuda.is_available(),
        fp16_opt_level="O1",
        
        # Configuration des sauvegardes
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Configuration de l'évaluation
        eval_steps=500,
        logging_steps=100,
    )
    
    # Configuration de la journalisation
    logging_config = LoggingConfig(
        wandb_project="neurolite",
        wandb_run_name=f"neurolite-{variant}",
        wandb_watch=False,
        wandb_log_model=False,
        log_level="info"
    )
    
    # Configuration des encodeurs multimodaux
    text_encoder_config = TextEncoderConfig(
        vocab_size=50000,
        hidden_size=config.get('hidden_size', 768),
        num_layers=config.get('num_hidden_layers', 12),
        dropout_rate=0.1,
        activation="gelu",
        layer_norm_eps=1e-6,
        max_position_embeddings=config.get('max_position_embeddings', 512),
        embedding_size=config.get('hidden_size', 768),
        use_learned_position_embeddings=True,
        use_relative_position_bias=True,
        share_embeddings_with_lm_head=True,
    )
    
    image_encoder_config = ImageEncoderConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_size=config.get('hidden_size', 768),
        num_layers=config.get('num_hidden_layers', 12),
        dropout_rate=0.1,
        activation="gelu",
        layer_norm_eps=1e-6,
        use_adaptive_patches=True,
        use_patch_dropout=True,
        patch_dropout_rate=0.1,
        use_cls_token=True,
    )
    
    # Configuration de l'encodeur audio
    audio_encoder_config = AudioEncoderConfig(
        sampling_rate=16000,
        n_mels=80,
        n_fft=400,
        hop_length=160,
        hidden_size=config.get('hidden_size', 768),
        num_layers=config.get('num_hidden_layers', 12),
        dropout_rate=0.1,
        activation="gelu",
        layer_norm_eps=1e-6,
        max_audio_length_ms=30000,
        use_spectrogram_augmentation=True
    )
    
    # Configuration de l'encodeur vidéo
    video_encoder_config = VideoEncoderConfig(
        num_frames=8,
        image_size=224,
        patch_size=16,
        temporal_patch_size=2,
        num_channels=3,
        hidden_size=config.get('hidden_size', 768),
        num_layers=config.get('num_hidden_layers', 12),
        dropout_rate=0.1,
        activation="gelu",
        layer_norm_eps=1e-6,
        use_temporal_attention=True,
        use_factorized_encoder=True,
        max_video_length_sec=30.0
    )
    
    # Configuration de l'encodeur de graphe
    graph_encoder_config = GraphEncoderConfig(
        node_feature_dim=128,
        edge_feature_dim=64,
        num_node_types=16,
        num_edge_types=8,
        hidden_size=config.get('hidden_size', 768),
        num_layers=config.get('num_hidden_layers', 12),
        dropout_rate=0.1,
        activation="gelu",
        layer_norm_eps=1e-6,
        use_graph_attention=True,
        num_graph_attention_heads=4,
        max_num_nodes=512,
        max_num_edges=2048
    )
    
    # Configuration du quantificateur
    quantizer_config = QuantizerConfig(
        n_embeddings=8192,
        embedding_dim=256,
        commitment_cost=0.25,
        use_ema_updates=True,
        ema_decay=0.99,
        restart_unused_codes=True,
        threshold_ema_dead_code=1e-5
    )
    
    # Mettre à jour la configuration du modèle avec les configurations des encodeurs
    model_config.text_encoder = text_encoder_config
    model_config.image_encoder = image_encoder_config
    model_config.audio_encoder = audio_encoder_config
    model_config.video_encoder = video_encoder_config
    model_config.graph_encoder = graph_encoder_config
    model_config.quantizer = quantizer_config
    
    # Mettre à jour les paramètres de génération dans la configuration d'entraînement
    training_config.generation_max_length = 512
    training_config.generation_min_length = 10
    training_config.generation_do_sample = True
    training_config.generation_early_stopping = True
    training_config.generation_num_beams = 5
    training_config.generation_temperature = 1.0
    training_config.generation_top_k = 50
    training_config.generation_top_p = 0.95
    training_config.generation_repetition_penalty = 1.2
    training_config.generation_length_penalty = 1.0
    training_config.generation_no_repeat_ngram_size = 3
    
    # Configuration pour la transformation des entrées
    model_config.use_bfloat16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    
    # Création de la configuration complète
    return NeuroLiteConfig(
        model_config=model_config,
        training_config=training_config,
        tokenizer_config=tokenizer_config,
        memory_config=memory_config,
        logging_config=logging_config,
        use_multimodal=config['use_multimodal_input'],
        modalities=["text", "image", "audio", "video", "graph"] if config['use_multimodal_input'] else ["text"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42
    )
    
def save_model(variant: str, output_dir: Path, force_cpu: bool = True) -> Dict[str, Any]:
    """
    Crée et sauvegarde un modèle NeuroLite de la variante spécifiée.
    
    Args:
        variant: La variante du modèle à créer (tiny, small, base, large)
        output_dir: Répertoire de sortie pour sauvegarder le modèle
        force_cpu: Forcer l'utilisation du CPU (True par défaut pour éviter les problèmes CUDA)
        
    Returns:
        Dict contenant les informations sur le modèle sauvegardé
    """
    print(f"\n=== Création du modèle {variant.upper()} ===\n")
    
    try:
        # Création de la configuration du modèle
        config = create_model_config(variant)
        
        # Désactiver la mémoire externe pour les modèles plus grands sur CPU
        if variant in ['small', 'base', 'large']:
            config.model_config.use_external_memory = False
            config.memory_config.use_external_memory = False
        
        # Forcer l'utilisation du CPU
        device = "cpu"
        config.device = device
        
        print(f"Configuration du modèle {variant.upper()}:")
        print(f"- Taille de couche cachée: {config.model_config.hidden_size}")
        print(f"- Nombre de couches: {config.model_config.num_hidden_layers}")
        print(f"- Têtes d'attention: {config.model_config.num_attention_heads}")
        print(f"- Taille intermédiaire: {config.model_config.intermediate_size}")
        print(f"- Mémoire externe: {'Activée' if config.model_config.use_external_memory else 'Désactivée'}")
        print(f"- Routage dynamique: {'Activé' if config.model_config.use_dynamic_routing else 'Désactivé'}")
        print(f"- Entrée multimodale: {'Activée' if config.use_multimodal else 'Désactivée'}")
        print(f"- Périphérique: {device}")
        
        print("\nInitialisation du modèle...\n")
        model = NeuroLiteModel(
            config=config,
            task_type="generation",
            num_labels=0  # Pour la génération de texte
        )
        
        # Afficher les informations du modèle
        model_size = get_model_size(model)
        print(f"\n=== Informations du modèle {variant.upper()} ===")
        print(f"- Paramètres totaux: {model_size['num_params']:,}")
        print(f"- Taille du modèle: {model_size['mb']:.2f} Mo ({model_size['gb']:.2f} Go)")
        
        # Créer le répertoire de sortie
        output_dir = output_dir / f"neurolite-{variant}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le modèle et la configuration
        print(f"\nSauvegarde du modèle dans {output_dir}...")
        model.save_pretrained(str(output_dir))
        config.save_pretrained(str(output_dir))
        
        # Créer un fichier README.md avec les informations du modèle
        create_readme(variant, output_dir, model_size, config)
        
        print(f"\n✅ Modèle {variant.upper()} sauvegardé avec succès dans {output_dir}")
        return model_size
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la création du modèle {variant}: {str(e)}")
        if hasattr(e, 'args') and e.args:
            print(f"Détails: {e.args}")
        import traceback
        traceback.print_exc()
        raise

def create_readme(variant: str, output_dir: Path, model_size: Dict[str, Any], config: NeuroLiteConfig) -> None:
    """
    Crée un fichier README.md avec les informations du modèle.
    
    Args:
        variant: Variante du modèle
        output_dir: Répertoire de sortie
        model_size: Dictionnaire contenant les informations de taille
        config: Configuration du modèle
    """
    readme_content = f"""# NeuroLite-{variant.upper()}

## Description
Modèle de langage NeuroLite de taille {variant} optimisé pour une exécution efficace sur différents matériels.

## Spécifications techniques

### Architecture
- **Type de modèle**: Transformer avec mémoire externe
- **Taille de couche cachée**: {config.model_config.hidden_size}
- **Nombre de couches**: {config.model_config.num_hidden_layers}
- **Têtes d'attention**: {config.model_config.num_attention_heads}
- **Taille intermédiaire**: {config.model_config.intermediate_size}
- **Paramètres totaux**: {model_size['num_params']:,}
- **Taille du modèle**: {model_size['mb']:.2f} Mo

### Fonctionnalités
- **Mémoire externe**: {'✅ Activée' if config.model_config.use_external_memory else '❌ Désactivée'}
- **Routage dynamique**: {'✅ Activé' if config.model_config.use_dynamic_routing else '❌ Désactivé'}
- **Entrée multimodale**: {'✅ Activée' if config.use_multimodal else '❌ Désactivée'}
- **Adaptation continue**: {'✅ Activée' if config.model_config.use_continual_adapter else '❌ Désactivée'}

## Utilisation

### Chargement du modèle

```python
from neurolite import NeuroLiteModel, NeuroLiteTokenizer

model = NeuroLiteModel.from_pretrained("username/neurolite-{variant}")
tokenizer = NeuroLiteTokenizer.from_pretrained("username/neurolite-{variant}")
```

### Génération de texte

```python
input_text = "Le langage est une capacité fondamentale de l'"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Entraînement

### Hyperparamètres recommandés
- Taille de batch: {config.training_config.per_device_train_batch_size}
- Taux d'apprentissage: {config.training_config.learning_rate}
- Poids de décroissance: {config.training_config.weight_decay}
- Pas d'échauffement: {config.training_config.warmup_steps}
- Nombre d'époques: {config.training_config.num_train_epochs}

## Limitations et biais

Comme tous les modèles de langage, NeuroLite peut générer du texte biaisé ou inexact. Il est important d'évaluer soigneusement les sorties du modèle avant de les utiliser en production.

## Licence

Ce modèle est distribué sous licence Apache 2.0.
"""
    
    # Écrire le fichier README.md
    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

def parse_args():
    """Parse les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(description="Génère différentes variantes de modèles NeuroLite")
    
    # Arguments principaux
    parser.add_argument(
        "--variants", 
        type=str, 
        nargs="+", 
        choices=['tiny', 'small', 'base', 'large', 'all'],
        default=['all'],
        help="Variantes de modèles à générer (par défaut: all)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=str(PROJECT_ROOT / "saved_models"),
        help="Répertoire de sortie pour les modèles générés"
    )
    
    # Options d'exécution
    parser.add_argument(
        "--cpu", 
        action="store_true", 
        help="Forcer l'utilisation du CPU même si un GPU est disponible"
    )
    parser.add_argument(
        "--skip-existing", 
        action="store_true", 
        help="Passer les modèles déjà générés"
    )
    
    # Options de débogage
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Activer le mode débogage (affiche plus d'informations)"
    )
    
    return parser.parse_args()

def main():
    """Fonction principale pour générer les modèles NeuroLite."""
    # Parser les arguments
    args = parse_args()
    
    # Configurer le mode débogage si nécessaire
    if args.debug:
        import logging
        logging.basicConfig(level=logging.INFO)
        logging.info("Mode débogage activé")
    
    # Configurer le périphérique
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        torch.set_default_tensor_type('torch.FloatTensor')
        print("\n⚠️  Utilisation du CPU forcée")
    
    # Déterminer les variantes à générer
    if 'all' in args.variants:
        variants = ['tiny', 'small', 'base', 'large']
    else:
        variants = list(set(args.variants))  # Supprimer les doublons
    
    # Créer le répertoire de sortie
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Génération des modèles NeuroLite ===")
    print(f"Répertoire de sortie: {output_dir}")
    print(f"Variantes à générer: {', '.join(variants)}")
    
    # Générer chaque variante
    sizes = {}
    for variant in variants:
        variant_dir = output_dir / f"neurolite-{variant}"
        
        # Vérifier si le modèle existe déjà
        if args.skip_existing and variant_dir.exists():
            print(f"\n⚠️  Le modèle {variant} existe déjà, passage au suivant...")
            continue
            
        try:
            print(f"\n{'=' * 50}")
            print(f"GÉNÉRATION DU MODÈLE: {variant.upper()}")
            print(f"{'=' * 50}")
            
            # Générer et sauvegarder le modèle
            size = save_model(variant, output_dir, force_cpu=args.cpu)
            if size:
                sizes[variant] = size
                
        except Exception as e:
            print(f"\n❌ Erreur lors de la génération du modèle {variant}: {str(e)}")
            if args.debug:
                import traceback
                traceback.print_exc()
    
    # Afficher un résumé si des modèles ont été générés
    if sizes:
        print("\n" + "=" * 50)
        print("RÉSUMÉ DES MODÈLES GÉNÉRÉS")
        print("=" * 50)
        
        for variant, size in sizes.items():
            print(f"\n{variant.upper():<8} - {size['num_params']:>12,} paramètres - {size['mb']:>7.1f} Mo")
        
        print("\n✅ Génération terminée avec succès!")
        
        # Afficher les instructions pour publier sur Hugging Face
        if len(sizes) > 0:
            print("\nPour publier les modèles sur Hugging Face, utilisez les commandes suivantes:")
            print("huggingface-cli login")
            for variant in sizes.keys():
                model_path = output_dir / f"neurolite-{variant}"
                print(f"huggingface-cli upload username/neurolite-{variant} {model_path}")
    else:
        print("\n❌ Aucun modèle n'a été généré.")

if __name__ == "__main__":
    main()
