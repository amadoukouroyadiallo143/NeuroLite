"""
Script de test pour les modules avancés de NeuroLite (Mémoire, Raisonnement, etc.).

Ce script vérifie que les modules complexes peuvent être activés, instanciés,
et qu'ils produisent une sortie lors d'une passe forward.
"""
import torch
import sys
import os

# Ajouter la racine du projet au sys.path pour les importations locales
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurolite.core.model import NeuroLiteModel
from neurolite.Configs.config import (
    NeuroLiteConfig, 
    ModelArchitectureConfig, 
    MMTextDecoderConfig,
    MemoryConfig,
    LongTermMemoryConfig,
    ReasoningConfig
)

def main():
    print("--- Test des Modules Avancés de NeuroLite ---")
    all_tests_passed = True

    try:
        # 1. Créer une configuration qui active tous les modules
        print("\n[1] Création d'une configuration avec modules avancés activés...")
        
        # Le modèle a besoin d'un vocab_size même si on ne s'en sert pas pour la logique ici
        vocab_size = 100 
        
        model_config = ModelArchitectureConfig(
            hidden_size=64,
            num_hidden_layers=2,
            # Activation des modules (flags principaux dans ModelArchitectureConfig)
            use_external_memory=True,
            use_hierarchical_memory=True,
            use_metacontroller=True, 
            # Configs spécifiques
            mm_text_decoder_config=MMTextDecoderConfig(vocab_size=vocab_size, input_dim=64)
        )

        reasoning_config = ReasoningConfig(
            use_symbolic_module=True,
            use_causal_reasoning=True,
            use_planning_module=False # On ne teste pas le planner pour l'instant
        )
        
        config = NeuroLiteConfig(
            model_config=model_config,
            reasoning_config=reasoning_config,
            # On passe aussi des configurations de mémoire pour être complet
            memory_config=MemoryConfig(use_external_memory=True, memory_dim=64),
            long_term_memory_config=LongTermMemoryConfig(enabled=True, dimension=64)
        )
        print("Configuration créée.")

        # 2. Instancier le modèle
        print("\n[2] Instanciation du modèle...")
        model = NeuroLiteModel(config, task_type="base")
        model.eval()
        print("Modèle instancié avec succès.")

        # 3. Préparer des données d'entrée factices
        print("\n[3] Préparation des données d'entrée factices...")
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        external_facts = torch.randn(batch_size, 3, model_config.hidden_size) # Faits externes pour le raisonnement
        task_info = {
            'causal_query': {
                'type': 'do',
                'intervention': {'X': 1} # Requête factice pour le moteur causal
            },
            'state_representation': torch.randn(batch_size, model_config.hidden_size) # État pour le metacontroller
        }
        print("Données d'entrée créées.")

        # 4. Exécuter la passe forward
        print("\n[4] Exécution de la passe forward avec les arguments avancés...")
        with torch.no_grad():
            outputs = model(
                multimodal_inputs={'input_ids': input_ids},
                update_memory=True,
                return_symbolic=True,
                external_facts=external_facts,
                task_info=task_info,
                return_dict=True
            )
        print("Passe forward terminée.")

        # 5. Vérifier les sorties
        print("\n[5] Vérification des sorties des modules...")
        
        # Test Mémoire
        if 'memory_output' in outputs and outputs['memory_output'] is not None:
            print("  [✓] Mémoire Externe : OK (sortie présente)")
        else:
            print("  [✗] Mémoire Externe : ERREUR (pas de sortie)")
            all_tests_passed = False

        # Test Raisonnement Symbolique
        if 'symbolic_output' in outputs and outputs['symbolic_output'] is not None:
            print("  [✓] Raisonnement Symbolique : OK (sortie présente)")
        else:
            print("  [✗] Raisonnement Symbolique : ERREUR (pas de sortie)")
            all_tests_passed = False

        # Test Moteur Causal
        if 'causal_output' in outputs and outputs['causal_output'] is not None:
            print("  [✓] Moteur Causal : OK (sortie présente)")
        else:
            print("  [✗] Moteur Causal : ERREUR (pas de sortie)")
            all_tests_passed = False
            
    except Exception as e:
        print(f"\n--- ERREUR CRITIQUE PENDANT LE TEST ---")
        import traceback
        traceback.print_exc()
        all_tests_passed = False

    # --- Conclusion ---
    if all_tests_passed:
        print("\n--- ✅ TOUS LES TESTS DE MODULES AVANCÉS SONT PASSÉS ---")
    else:
        print("\n--- ❌ CERTAINS TESTS DE MODULES AVANCÉS ONT ÉCHOUÉ ---")

if __name__ == "__main__":
    main() 