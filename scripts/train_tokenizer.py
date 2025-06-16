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
import time

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
        default=32000, 
        help="Taille du vocabulaire à construire."
    )
    parser.add_argument(
        "--max_train_samples", 
        type=int, 
        default=0, 
        help="Nombre max d'exemples pour l'entraînement du tokenizer (0 pour tous)."
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
    script_start_time = time.time()
    
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
    
    # Appliquer le filtre max_train_samples si spécifié
    if args.max_train_samples > 0:
        if len(dataset) > args.max_train_samples:
            print(f"Utilisation d'un sous-ensemble de {args.max_train_samples} exemples pour l'entraînement du tokenizer.")
            dataset = dataset.select(range(args.max_train_samples))
        else:
            print(f"Le nombre d'échantillons demandé ({args.max_train_samples}) est supérieur ou égal au nombre total d'exemples ({len(dataset)}). Utilisation du dataset complet.")

    # 3. Configurer et initialiser le tokenizer
    print("Initialisation d'un nouveau tokenizer...")
    # La taille du vocabulaire est maintenant contrôlée par la ligne de commande.
    # Le tokenizer a besoin d'une config minimale pour s'initialiser.
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
    
    # --- Mesure du temps d'entraînement ---
    train_start_time = time.time()
    # La méthode build_vocab gère l'entraînement BPE
    tokenizer.build_vocab(dataset, text_column='text', chat_format=True)
    train_elapsed_seconds = time.time() - train_start_time
    # --- Fin de la mesure ---
    
    # 5. Sauvegarder le tokenizer
    print(f"Entraînement terminé. Sauvegarde du tokenizer dans : {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_path))
    
    script_elapsed_seconds = time.time() - script_start_time

    # --- Affichage des statistiques ---
    print("\n" + "="*50)
    print("Statistiques de l'entraînement du Tokenizer")
    print("="*50)
    print(f"Temps total d'exécution     : {script_elapsed_seconds:.2f} secondes")
    print(f"Temps d'entraînement BPE pur : {train_elapsed_seconds:.2f} secondes")
    
    final_vocab_size = tokenizer.vocab_size
    print(f"Taille du vocabulaire finale : {final_vocab_size} (Cible: {args.vocab_size})")

    if final_vocab_size < args.vocab_size:
        print("NOTE: La taille finale est inférieure à la cible. C'est normal si le corpus de texte est trop petit ou peu varié.")
    print("="*50)
    
    print(f"\nOpération terminée avec succès.")
    print(f"Le tokenizer est prêt à être utilisé depuis le dossier : '{output_path}'")

if __name__ == "__main__":
    main() 