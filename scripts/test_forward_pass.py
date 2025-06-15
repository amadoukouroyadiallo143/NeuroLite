"""
Script de test pour la passe forward du modèle NeuroLite.

Ce script effectue les actions suivantes :
1.  Charge un tokenizer pré-entraîné.
2.  Crée une configuration minimale pour NeuroLite.
3.  Instancie le modèle NeuroLite.
4.  Prépare des données d'entrée factices (dummy data).
5.  Exécute une passe forward.
6.  Affiche la forme des sorties pour vérifier que le flux de données est correct.
"""

import torch
from transformers import AutoTokenizer
import sys
import os

# Ajouter la racine du projet au sys.path pour les importations locales
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurolite.core.model import NeuroLiteModel
from neurolite.Configs.config import NeuroLiteConfig, ModelArchitectureConfig, MMTextDecoderConfig, TokenizerConfig

def main():
    """Fonction principale du script de test."""
    print("--- Test de la Passe Forward de NeuroLite ---")

    try:
        # 1. Charger le tokenizer
        print("Chargement du tokenizer 'gpt2'...")
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size
        print(f"Tokenizer chargé. Taille du vocabulaire: {vocab_size}")

        # 2. Créer une configuration minimale (tiny)
        print("Création d'une configuration de modèle 'tiny'...")
        model_config = ModelArchitectureConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            max_seq_length=128,
            mm_text_decoder_config=MMTextDecoderConfig(vocab_size=vocab_size, input_dim=64)
        )
        config = NeuroLiteConfig(
            model_config=model_config,
            tokenizer_config=TokenizerConfig(hidden_size=64)
        )
        print("Configuration créée.")

        # 3. Instancier le modèle
        print("Instanciation du modèle...")
        model = NeuroLiteModel(config, task_type='generation')
        model.eval()  # Mettre le modèle en mode évaluation
        print("Modèle instancié avec succès.")

        # 4. Préparer des données factices
        print("Préparation des données d'entrée factices...")
        dummy_texts = [
            "Ceci est une phrase de test.",
            "Voici une autre phrase un peu plus longue pour voir.",
            "Test."
        ]
        inputs = tokenizer(dummy_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        print(f"Données créées. Shape des input_ids: {input_ids.shape}")

        # 5. Exécuter la passe forward
        print("Exécution de la passe forward...")
        with torch.no_grad():
            outputs = model(
                multimodal_inputs={'input_ids': input_ids}, 
                attention_mask=attention_mask,
                return_dict=True
            )
        print("Passe forward terminée.")

        # 6. Vérifier les sorties
        print("Vérification des sorties...")
        if 'logits' in outputs and outputs['logits'] is not None:
            logits = outputs['logits']
            print(f"  - Type des logits: {type(logits)}")
            print(f"  - Shape des logits: {logits.shape}")
            
            # Vérifier que la shape est correcte : (batch_size, seq_len, vocab_size)
            expected_shape = (input_ids.shape[0], input_ids.shape[1], vocab_size)
            if logits.shape == expected_shape:
                print("  - La shape des logits est CORRECTE.")
            else:
                print(f"  - ERREUR: La shape des logits est INCORRECTE. Attendu: {expected_shape}, Obtenu: {logits.shape}")
        else:
            print("  - ERREUR: Pas de 'logits' trouvés dans la sortie du modèle.")
            print(f"  - Sorties obtenues: {outputs.keys()}")

        print("\n--- Test terminé avec succès ! ---")

    except Exception as e:
        print(f"\n--- ERREUR LORS DU TEST DE LA PASSE FORWARD ---")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 