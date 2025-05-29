#!/usr/bin/env python3
"""
Script de débogage pour identifier les appels à CUDA dans NeuroLite.
"""

import os
import sys
import torch
from pathlib import Path

# Désactiver complètement CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Forcer PyTorch à utiliser le CPU
torch.set_default_tensor_type('torch.FloatTensor')

# Remplacer les fonctions CUDA pour intercepter les appels
def cuda_error(*args, **kwargs):
    raise RuntimeError("CUDA est désactivé. Tentative d'utilisation de CUDA détectée.")

# Remplacer les fonctions CUDA
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_name = lambda x: 'cpu'
torch.cuda.get_device_capability = lambda x: (0, 0)
torch.cuda.set_device = cuda_error
torch.cuda.device = cuda_error
torch.cuda.device_of = cuda_error
torch.cuda.stream = cuda_error

# Ajouter le répertoire parent au PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

print("=== Configuration du débogage CUDA ===")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Device par défaut: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
print(f"Type de tenseur par défaut: {torch.get_default_dtype()}")

# Importer les composants de NeuroLite après avoir configuré l'environnement
print("\n=== Importation des modules NeuroLite ===")
try:
    from neurolite.Configs.config import NeuroLiteConfig, ModelArchitectureConfig, TrainingConfig
    from neurolite.core.model import NeuroLiteModel
    print("Importation réussie des modules NeuroLite")
except Exception as e:
    print(f"Erreur lors de l'importation: {e}")
    sys.exit(1)

# Configuration minimale
def create_minimal_config():
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
    
    return config

# Tester l'initialisation du modèle
def test_model_initialization():
    print("\n=== Test d'initialisation du modèle ===")
    try:
        config = create_minimal_config()
        print("Configuration créée avec succès")
        
        # Vérifier la configuration
        print("\nVérification de la configuration:")
        print(f"Device dans config: {config.device}")
        if hasattr(config, 'model_config'):
            print(f"Device dans model_config: {getattr(config.model_config, 'device', 'non défini')}")
        
        # Essayer d'initialiser le modèle
        print("\nTentative d'initialisation du modèle...")
        model = NeuroLiteModel(
            config=config,
            task_type="generation",
            num_labels=2
        )
        
        print("Modèle initialisé avec succès!")
        return model
        
    except Exception as e:
        print(f"Erreur lors de l'initialisation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    model = test_model_initialization()
    if model is not None:
        print("\n=== Test réussi! Le modèle a été initialisé sur le CPU. ===")
    else:
        print("\n=== Échec de l'initialisation du modèle. Voir les erreurs ci-dessus. ===")
        sys.exit(1)
