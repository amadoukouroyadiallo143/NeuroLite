"""
Script dédié à l'entraînement d'un nouveau tokenizer NeuroLite.

Ce script charge un jeu de données, entraîne un tokenizer BPE à partir de zéro,
et le sauvegarde dans un répertoire spécifié pour une utilisation ultérieure
par le script d'entraînement principal.
"""
import argparse
import sys
import os
from pathlib import Path

from datasets import load_from_disk

# Ajouter la racine du projet au sys.path pour les importations locales
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurolite.tokenization.tokenizer import NeuroLiteTokenizer
from neurolite.Configs.config import NeuroLiteConfig, TokenizerConfig

def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Entraînement d'un tokenizer NeuroLite")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="data/raw/mixture_of_thoughts", 
        help="Chemin vers le dataset brut (doit contenir un split 'train')."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./outputs/neurolite/neurolite_tokenizer", 
        help="Répertoire où sauvegarder le tokenizer entraîné."
    )
    parser.add_argument(
        "--vocab_size", 
        type=int, 
        default=50000, 
        help="Taille du vocabulaire à construire."
    )
    parser.add_argument(
        "--hidden_size", 
        type=int, 
        default=512, 
        help="Dimension cachée requise par la configuration du tokenizer."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    output_path = Path(args.output_dir)
    dataset_path = Path(args.dataset_path)

    # 1. Vérifier si le tokenizer existe déjà
    if output_path.exists() and (output_path / "tokenizer_config.json").exists():
        print(f"Un tokenizer semble déjà exister dans : {output_path}")
        print("Entraînement ignoré. Supprimez le dossier pour forcer un nouvel entraînement.")
        return

    # 2. Charger le jeu de données
    # Le tokenizer doit être entraîné sur du texte brut, de préférence un grand corpus.
    # On charge la partie 'train' du dataset.
    train_dataset_path = dataset_path / "train"
    if not train_dataset_path.exists():
        raise FileNotFoundError(
            f"Le split d'entraînement n'a pas été trouvé dans {dataset_path}. "
            f"Assurez-vous que le chemin est correct et que le dataset a été téléchargé."
        )
        
    print(f"Chargement du dataset depuis {train_dataset_path} pour l'entraînement du tokenizer...")
    dataset = load_from_disk(str(train_dataset_path))
    
    # 3. Configurer et initialiser le tokenizer
    print("Initialisation d'un nouveau tokenizer...")
    # Le tokenizer a besoin d'une config minimale pour s'initialiser
    tokenizer_config = TokenizerConfig(
        vocab_size=args.vocab_size, 
        hidden_size=args.hidden_size
    )
    # L'objet NeuroLiteConfig est nécessaire pour la compatibilité
    config = NeuroLiteConfig(tokenizer_config=tokenizer_config)

    tokenizer = NeuroLiteTokenizer(tokenizer_config, neurolite_config=config)

    # 4. Entraîner le tokenizer
    print(f"Début de l'entraînement du tokenizer sur {len(dataset)} exemples...")
    print(f"Taille du vocabulaire cible : {args.vocab_size}")
    
    # La méthode build_vocab gère l'entraînement BPE
    tokenizer.build_vocab(dataset, text_column='text', chat_format=True)
    
    # 5. Sauvegarder le tokenizer
    print(f"Entraînement terminé. Sauvegarde du tokenizer dans : {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_path))
    
    print("\nOpération terminée avec succès.")
    print(f"Le tokenizer est prêt à être utilisé depuis le dossier : '{output_path}'")

if __name__ == "__main__":
    main() 