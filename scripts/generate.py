"""
Script pour la génération de texte avec un modèle NeuroLite entraîné.

Ce script charge un modèle depuis un checkpoint et vous permet d'interagir
avec lui en lui donnant des prompts pour générer des réponses.
"""

import torch
import argparse
import os
import sys

# Ajouter la racine du projet au sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurolite.core.model import NeuroLiteModel

def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Génération de texte avec NeuroLite")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Chemin vers le répertoire du modèle entraîné."
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=100, 
        help="Nombre maximum de nouveaux tokens à générer."
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.8, 
        help="Contrôle le caractère aléatoire. Plus la valeur est élevée, plus c'est aléatoire."
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=50, 
        help="Ne considère que les 'k' tokens les plus probables pour la génération."
    )
    return parser.parse_args()

@torch.no_grad()
def generate():
    """Fonction principale de génération."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Chargement du modèle depuis : {args.model_path}")
    model = NeuroLiteModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    print("\nModèle NeuroLite chargé. Prêt pour la génération.")
    print("Entrez votre prompt. Tapez 'exit' ou 'quit' pour terminer.")
    print("-" * 50)

    while True:
        prompt_text = input("Prompt > ")
        if prompt_text.lower() in ["exit", "quit"]:
            break

        # Préparation des entrées pour le modèle
        # Notre modèle attend un dictionnaire multimodal
        multimodal_inputs = {
            'text': [prompt_text] # Le tokenizer interne s'attend à une liste de textes
        }
        
        print("Génération en cours...")
        
        # Appel de la méthode de génération du modèle
        outputs = model.generate(
            multimodal_inputs=multimodal_inputs,
            target_modalities=['text'], # Nous voulons générer du texte
            max_length=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            do_sample=True # Activer l'échantillonnage pour la créativité
        )
        
        generated_text = outputs.get('text', ["Erreur lors de la génération."])[0]

        print("-" * 50)
        print("Réponse du modèle:")
        print(prompt_text + generated_text)
        print("-" * 50)

if __name__ == "__main__":
    generate() 