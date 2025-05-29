#!/usr/bin/env python3
"""
Script d'exemple pour initialiser et sauvegarder un modèle NeuroLite.

Ce script montre comment :
1. Charger une configuration prédéfinie
2. Initialiser un modèle NeuroLite
3. Sauvegarder le modèle
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple

# Forcer l'utilisation du CPU
device = torch.device('cpu')
print(f"Utilisation du dispositif: {device}")

# Afficher les informations sur CUDA
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Version PyTorch: {torch.__version__}")

# Assurez-vous que le chemin racine du projet est dans le PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.append(str(PROJECT_ROOT))

from neurolite.core.model import NeuroLiteModel
from neurolite.Configs.config import NeuroLiteConfig, ModelArchitectureConfig, TrainingConfig

def get_model_size(model: torch.nn.Module) -> Dict[str, Union[str, int, float]]:
    """
    Calcule et retourne la taille du modèle en octets, Mo et Go.
    
    Args:
        model: Modèle PyTorch
        
    Returns:
        Dictionnaire contenant la taille en octets, Mo et Go
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    size_all_gb = size_all_mb / 1024
    
    return {
        'bytes': param_size + buffer_size,
        'mb': size_all_mb,
        'gb': size_all_gb,
        'num_params': sum(p.numel() for p in model.parameters())
    }

def main():
    # Désactiver complètement CUDA
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Forcer PyTorch à utiliser le CPU
    import torch
    torch.set_default_tensor_type('torch.FloatTensor')
    
    # Remplacer les fonctions CUDA pour intercepter les appels
    def cuda_error(*args, **kwargs):
        raise RuntimeError("CUDA est désactivé. Tentative d'utilisation de CUDA détectée.")
    
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_name = lambda x: 'cpu'
    torch.cuda.get_device_capability = lambda x: (0, 0)
    torch.cuda.set_device = cuda_error
    torch.cuda.device = cuda_error
    torch.cuda.device_of = cuda_error
    torch.cuda.stream = cuda_error
    
    print("=== Configuration du device ===")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    print(f"Device par défaut: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Créer une configuration minimale qui fonctionne avec le CPU
    model_config = ModelArchitectureConfig(
        hidden_size=64,  # Taille réduite pour le débogage
        num_hidden_layers=2,
        input_projection_type="minhash_bloom",
        minhash_num_permutations=32,  # Réduit pour le débogage
        bloom_filter_size=128,  # Réduit pour le débogage
        dropout_rate=0.1,
        vocab_size=1000,  # Réduit pour le débogage
        num_attention_heads=2,
        intermediate_size=128,  # Réduit pour le débogage
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=128,  # Réduit pour le débogage
        use_external_memory=False,  # Désactiver la mémoire externe pour simplifier
        use_dynamic_routing=False,  # Désactiver le routage dynamique pour simplifier
        use_multimodal_input=False,  # Désactiver l'entrée multimodale pour simplifier
    )
    
    training_config = TrainingConfig(
        output_dir=str(PROJECT_ROOT / "models"),
        train_data_path="data/processed/train",
        val_data_path="data/processed/val"
    )
    
    config = NeuroLiteConfig(
        model_config=model_config,
        training_config=training_config
    )
    
    # Forcer l'utilisation du CPU
    config.device = "cpu"
    if hasattr(config, 'model_config'):
        config.model_config.device = "cpu"
    
    try:
        # Initialiser le modèle
        print("Initialisation du modèle NeuroLite...")
        try:
            # Forcer le modèle à utiliser le CPU
            model = NeuroLiteModel(
                config=config,
                task_type="generation",  # Options: "generation", "classification", "sequence_labeling"
                num_labels=2  # Requis pour la classification
            ).to(device)  # Déplacer le modèle sur le CPU
            print("Modèle initialisé avec succès sur le CPU")
            
            # Afficher la taille du modèle
            model_size = get_model_size(model)
            print(f"\n=== Taille du modèle ===")
            print(f"Paramètres: {model_size['num_params']:,}")
            print(f"Mémoire utilisée: {model_size['mb']:.2f} Mo ({model_size['gb']:.2f} Go)")
            
        except Exception as e:
            print(f"Erreur lors de l'initialisation du modèle: {str(e)}")
            raise
        
        # Créer le répertoire de sortie
        output_dir = PROJECT_ROOT / "saved_models" / "neurolite_base"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le modèle
        print(f"Sauvegarde du modèle dans {output_dir}...")
        model.save_pretrained(str(output_dir))
        
        # Sauvegarder également la configuration
        config.save_pretrained(str(output_dir))
        
        print(f"✅ Modèle et configuration sauvegardés avec succès dans {output_dir}")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation ou de la sauvegarde du modèle: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# Pour charger le modèle plus tard
# from transformers import AutoModel
# model = AutoModel.from_pretrained(output_dir)