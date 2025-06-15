"""
Script pour évaluer un modèle NeuroLite entraîné.

Ce script charge un modèle depuis un checkpoint, le fait tourner sur un jeu
de données d'évaluation et calcule des métriques de performance comme la perplexité.
"""

import torch
import argparse
import os
import sys
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset
from pathlib import Path

# Ajouter la racine du projet au sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurolite.core.model import NeuroLiteModel
from neurolite.training.data_collator import SFTDataCollator

def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Évaluation d'un modèle NeuroLite")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Chemin vers le répertoire du modèle entraîné (contenant config.json et les poids)."
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="data/raw/mixture_of_thoughts/test", 
        help="Chemin vers le dataset d'évaluation. Si non trouvé, essaiera de se replier sur le split 'train'."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="Taille du batch pour l'évaluation."
    )
    parser.add_argument(
        "--max_eval_samples", 
        type=int, 
        default=500, 
        help="Nombre max d'exemples d'évaluation (0 pour tous, 500 par défaut pour un test rapide)."
    )
    return parser.parse_args()

def get_device():
    """Détermine et retourne le device à utiliser (CUDA ou CPU)."""
    if torch.cuda.is_available():
        print("Utilisation du device : CUDA")
        return torch.device("cuda")
    else:
        print("Utilisation du device : CPU")
        return torch.device("cpu")

def load_evaluation_dataset(dataset_path_str: str, max_samples: int) -> Dataset:
    """Charge le dataset d'évaluation depuis le disque, avec une logique de repli sur le split 'train'."""
    print(f"\nChargement du dataset d'évaluation...")
    dataset_path = Path(dataset_path_str)
    
    if not dataset_path.exists():
        print(f"INFO: Le chemin spécifié '{dataset_path}' n'existe pas.")
        # Logique de repli: si le chemin se termine par 'test', on essaie 'train'
        if dataset_path.name == 'test':
            train_path = dataset_path.parent / 'train'
            print(f"INFO: Tentative de repli sur le split d'entraînement : '{train_path}'")
            if train_path.exists():
                print("\n" + "="*70)
                print("AVERTISSEMENT: Le split de test n'a pas été trouvé.")
                print(f"Utilisation du split d'entraînement '{train_path}' pour l'évaluation.")
                print("Pour une évaluation rigoureuse, veuillez créer un split de test dédié.")
                print("="*70 + "\n")
                dataset_path = train_path
            else:
                print("\n" + "="*70)
                print(f"ERREUR: Aucun dataset trouvé.")
                print(f"  - Chemin de test (échoué) : '{dataset_path}'")
                print(f"  - Chemin de train (échoué): '{train_path}'")
                print("L'évaluation ne peut pas continuer. Veuillez vérifier les chemins.")
                print("="*70 + "\n")
                sys.exit(1)
        else:
            # Si le chemin spécifié n'est pas 'test' et n'existe pas, c'est une erreur fatale.
            print("\n" + "="*70)
            print(f"ERREUR: Le chemin de dataset spécifié n'a pas été trouvé : '{dataset_path}'")
            print("L'évaluation ne peut pas continuer.")
            print("="*70 + "\n")
            sys.exit(1)
    
    print(f"Chargement des données depuis : '{dataset_path}'")
    dataset = load_from_disk(str(dataset_path))
    
    if max_samples > 0 and len(dataset) > max_samples:
        print(f"Sélection d'un sous-ensemble de {max_samples} exemples pour l'évaluation.")
        dataset = dataset.select(range(max_samples))
        
    print(f"Évaluation sur {len(dataset)} exemples.")
    return dataset

@torch.no_grad()
def evaluate():
    """Fonction principale d'évaluation."""
    args = parse_args()
    device = get_device()

    # --- 1. Chargement du Modèle ---
    print(f"\nChargement du modèle depuis : {args.model_path}")
    try:
        model = NeuroLiteModel.from_pretrained(args.model_path, task_type='multimodal_generation')
        model.to(device)
        model.eval()
        print("Modèle chargé avec succès.")
    except Exception as e:
        print(f"\nERREUR: Impossible de charger le modèle depuis {args.model_path}.")
        print(f"Détail de l'erreur: {e}")
        sys.exit(1)

    tokenizer = model.tokenizer
    config = model.config

    # --- 2. Chargement des Données ---
    eval_dataset = load_evaluation_dataset(args.dataset_path, args.max_eval_samples)

    # --- 3. Préparation du DataLoader ---
    data_collator = SFTDataCollator(
        tokenizer=tokenizer,
        max_length=config.model_config.max_seq_length
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=False
    )

    # --- 4. Boucle d'Évaluation ---
    total_loss = 0.0
    total_tokens = 0

    print("\nDébut de la boucle d'évaluation...")
    for batch in tqdm(eval_dataloader, desc="Évaluation"):
        try:
            multimodal_inputs = {
                'text': {
                    'input_ids': batch['multimodal_inputs']['text']['input_ids'].to(device),
                    'attention_mask': batch['multimodal_inputs']['text']['attention_mask'].to(device)
                }
            }
            labels = batch['labels'].to(device)
        except (KeyError, AttributeError) as e:
            print(f"\nERREUR: Le batch du dataloader a une structure inattendue: {e}")
            continue

        outputs = model(multimodal_inputs=multimodal_inputs, labels=labels)
        loss = outputs.get('loss')
        
        if loss is not None:
            num_tokens = (labels != -100).sum().item()
            if num_tokens > 0:
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

    # --- 5. Calcul et Affichage des Résultats ---
    if total_tokens > 0:
        mean_loss = total_loss / total_tokens
        perplexity = np.exp(mean_loss)
    else:
        print("\nAVERTISSEMENT: Aucun token valide trouvé pour le calcul de la perte.")
        mean_loss = float('inf')
        perplexity = float('inf')

    print("\n" + "="*40)
    print("--- RÉSULTATS DE L'ÉVALUATION ---")
    print(f"  - Perte moyenne    : {mean_loss:.4f}")
    print(f"  - Perplexité       : {perplexity:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    evaluate() 