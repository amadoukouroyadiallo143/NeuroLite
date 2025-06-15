"""
Script pour tester un modèle NeuroLite entraîné en générant du texte.
"""
import torch
import argparse
import sys
import os

# Ajouter la racine du projet au sys.path pour les importations locales
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurolite.core.model import NeuroLiteModel

def main():
    parser = argparse.ArgumentParser(description="Génération de texte avec un modèle NeuroLite entraîné.")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin vers le répertoire du modèle sauvegardé.")
    parser.add_argument("--prompt", type=str, required=True, help="Phrase de départ pour la génération.")
    parser.add_argument("--max_length", type=int, default=100, help="Longueur maximale du texte généré.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Contrôle le caractère aléatoire de la génération.")
    parser.add_argument("--top_k", type=int, default=50, help="Filtre les tokens les moins probables.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Échantillonnage Nucleus.")
    parser.add_argument("--greedy", action='store_true', help="Utilise la génération gloutonne (greedy search) au lieu de l'échantillonnage.")
    # --- Arguments pour les modules avancés ---
    parser.add_argument("--use_memory", action='store_true', help="Active l'utilisation de la mémoire pendant la génération.")
    parser.add_argument("--continuous_learning", action='store_true', help="Active l'apprentissage continu pendant la génération.")
    parser.add_argument("--return_symbolic", action='store_true', help="Tente de retourner une sortie symbolique (si le modèle le supporte).")
    args = parser.parse_args()

    print(f"Chargement du modèle depuis : {args.model_path}")

    # Déterminer le device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du device : {device}")

    try:
        # Charger le modèle et le tokenizer directement avec from_pretrained
        # Cette méthode s'occupe de charger la config, le tokenizer, et les poids.
        model = NeuroLiteModel.from_pretrained(args.model_path, task_type='generation')
        model.to(device)
        model.eval()

        print("Modèle chargé avec succès.")

        # La méthode generate gère maintenant l'encodage en interne.
        # On passe directement le texte brut.
        multimodal_inputs = {'text': [args.prompt]} # Doit être une liste

        # Générer le texte
        print("\n--- Génération du texte ---")
        print(f"Prompt: {args.prompt}")
        
        # Préparer les arguments pour la méthode generate
        generation_kwargs = {
            'update_memory': args.use_memory,
            'continuous_learning': args.continuous_learning,
            'return_symbolic': args.return_symbolic,
            'max_length': args.max_length,
            'temperature': args.temperature,
            'top_k': args.top_k,
            'top_p': args.top_p,
            'do_sample': not args.greedy,
            # On specifie la modalité de sortie pour obtenir du texte décodé
            'target_modalities': ['text'] 
        }

        with torch.no_grad():
            # La sortie est un dictionnaire contenant le texte généré
            output = model.generate(
                multimodal_inputs=multimodal_inputs,
                **generation_kwargs
            )

        # Décoder et afficher la sortie
        # La sortie est un dictionnaire avec la clé de la modalité, ex: {'text': ['texte généré']}
        generated_text = output.get('text', ["Impossible de générer du texte."])[0]
        
        print("\n--- Résultat ---")
        print(generated_text)
        print("----------------\n")

    except Exception as e:
        print(f"\n--- ERREUR LORS DE LA GÉNÉRATION ---")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 