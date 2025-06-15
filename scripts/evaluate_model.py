"""
Script pour évaluer un modèle NeuroLite entraîné sur un jeu de données de test.
Calcule la perte (loss) et la perplexité.
"""
import torch
from datasets import load_from_disk
import argparse
import sys
import os
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

# Ajouter la racine du projet au sys.path pour les importations locales
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurolite.core.model import NeuroLiteModel
from neurolite.training.data_collator import SFTDataCollator

def preprocess_sft_dataset(dataset):
    """
    Formate le dataset pour avoir des colonnes 'prompt' et 'response' claires.
    """
    def format_example(example):
        prompt_texts = []
        response_texts = []
        # Gère le format de conversation où les messages sont dans une liste
        if 'messages' in example and isinstance(example['messages'], list):
            for message in example['messages']:
                if message.get('role') == 'user':
                    prompt_texts.append(message.get('content', ''))
                elif message.get('role') == 'assistant':
                    response_texts.append(message.get('content', ''))
            prompt = "\n".join(filter(None, prompt_texts))
            response = "\n".join(filter(None, response_texts))
            return {'prompt': prompt, 'response': response}
        # Gère un format déjà plat
        return {'prompt': example.get('prompt', ''), 'response': example.get('response', '')}

    # Appliquer le formatage et supprimer les anciennes colonnes si elles existent
    column_names = dataset.column_names
    return dataset.map(
        format_example,
        remove_columns=[name for name in column_names if name not in ['prompt', 'response']]
    )

def move_to_device(batch, device):
    """Déplace récursivement les tenseurs d'un batch vers le device spécifié."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    else:
        return batch

def main():
    parser = argparse.ArgumentParser(description="Évaluation d'un modèle NeuroLite.")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin vers le répertoire du modèle sauvegardé.")
    parser.add_argument("--dataset_path", type=str, default="data/raw/mixture_of_thoughts", help="Chemin vers le dataset d'évaluation complet (contenant le split 'train').")
    parser.add_argument("--batch_size", type=int, default=4, help="Taille du batch pour l'évaluation.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Nombre maximum d'exemples à évaluer (None pour tous).")
    args = parser.parse_args()

    print(f"Évaluation du modèle : {args.model_path}")

    # --- 1. Chargement ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du device : {device}")

    try:
        # Charger le modèle et le tokenizer propriétaire avec la méthode unifiée
        model = NeuroLiteModel.from_pretrained(args.model_path, task_type='multimodal_generation')
        tokenizer = model.tokenizer
        
        model.to(device)
        model.eval()

        print("Modèle et tokenizer chargés avec succès.")

        # --- 2. Préparation des données ---
        print(f"Chargement du dataset depuis {args.dataset_path}...")
        dataset_path = Path(args.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset non trouvé à : {dataset_path}")
        
        # On charge le split 'train' et on en tire un set d'évaluation déterministe
        full_dataset = load_from_disk(os.path.join(dataset_path, 'train'))
        eval_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)['test']
        
        print("Prétraitement du dataset d'évaluation...")
        eval_dataset = preprocess_sft_dataset(eval_dataset)

        if args.max_eval_samples and args.max_eval_samples < len(eval_dataset):
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        
        print(f"Évaluation sur {len(eval_dataset)} exemples.")

        data_collator = SFTDataCollator(tokenizer=tokenizer, max_length=model.config.model_config.max_seq_length)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=False)

        # --- 3. Boucle d'évaluation ---
        print("Début de la boucle d'évaluation...")
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Évaluation"):
                # Déplacer le batch (potentiellement imbriqué) sur le bon device
                inputs = move_to_device(batch, device)
                
                outputs = model(**inputs)
                
                loss = outputs.get('loss')
                if loss is not None:
                    # La taille du batch se trouve dans la structure d'entrée multimodale
                    batch_size = inputs['multimodal_inputs']['text']['input_ids'].size(0)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size

        # --- 4. Calcul et affichage des métriques ---
        if total_samples == 0:
            print("Aucun échantillon n'a été évalué ou aucune perte n'a été calculée.")
            return

        avg_loss = total_loss / total_samples
        perplexity = math.exp(avg_loss)

        print("\n--- Métriques d'Évaluation ---")
        print(f"  Perte (Loss) moyenne : {avg_loss:.4f}")
        print(f"  Perplexité (Perplexity) : {perplexity:.4f}")
        print("------------------------------\n")

    except Exception as e:
        print(f"\n--- ERREUR LORS DE L'ÉVALUATION ---")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 