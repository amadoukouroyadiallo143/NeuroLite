"""
Script pour explorer le dataset Mixture of Thoughts
"""

import os
import json
from datasets import load_from_disk
import pandas as pd
from pathlib import Path
import pprint

# Chemin vers le dataset
dataset_path = Path("data/raw/mixture_of_thoughts/train")

# Charger le dataset
print("Chargement du dataset...")
dataset = load_from_disk(dataset_path)

# Afficher les informations de base
print("\nInformations sur le dataset:")
print(f"Nombre total d'exemples: {len(dataset)}")
print(f"Features: {dataset.features}")
print(f"Column names: {dataset.column_names}")

# Afficher quelques exemples
print("\nPremiers exemples:")
for i in range(min(3, len(dataset))):
    print(f"\nExemple {i+1}:")
    example = dataset[i]
    
    # Afficher les informations brutes pour comprendre la structure
    print("Structure brute de l'exemple:")
    print(f"Type de l'exemple: {type(example)}")
    print(f"Clés dans l'exemple: {list(example.keys()) if hasattr(example, 'keys') else 'Pas un dictionnaire'}")
    
    # Analyser chaque champ séparément
    if 'messages' in example:
        print("\nChamp 'messages':")
        messages = example['messages']
        print(f"Type de messages: {type(messages)}")
        print(f"Nombre d'éléments: {len(messages) if hasattr(messages, '__len__') else 'N/A'}")
        
        # Examiner quelques messages s'ils existent
        if isinstance(messages, list) and len(messages) > 0:
            for j, msg in enumerate(messages[:3]):  # Afficher jusqu'à 3 messages
                print(f"  Message {j+1}:")
                if isinstance(msg, dict):
                    for k, v in msg.items():
                        if isinstance(v, str) and len(v) > 200:
                            print(f"    {k}: {v[:200]}... (tronqué)")
                        else:
                            print(f"    {k}: {v}")
                else:
                    print(f"    {type(msg)}: {str(msg)[:100]}... (tronqué)" if isinstance(msg, str) and len(str(msg)) > 100 else f"    {msg}")
    
    # Afficher les autres champs
    for key in example.keys() if hasattr(example, 'keys') else []:
        if key != 'messages':
            value = example[key]
            if isinstance(value, str) and len(value) > 200:
                print(f"\nChamp '{key}': {value[:200]}... (tronqué)")
            else:
                print(f"\nChamp '{key}': {value}")

# Calculer la distribution des sources si disponible
if 'source' in dataset.column_names:
    try:
        sources = {}
        for i in range(min(1000, len(dataset))):
            source = dataset[i]['source']
            sources[source] = sources.get(source, 0) + 1
        
        print("\nDistribution des sources (premiers 1000 exemples):")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            print(f"{source}: {count} exemples ({count/10:.1f}%)")
    except Exception as e:
        print(f"\nErreur lors du calcul des sources: {e}")

# Statistiques sur les tokens si disponible
if 'num_tokens' in dataset.column_names:
    try:
        token_counts = [dataset[i]['num_tokens'] for i in range(min(1000, len(dataset)))]
        print("\nStatistiques sur les nombres de tokens (premiers 1000 exemples):")
        print(f"Minimum: {min(token_counts)}")
        print(f"Maximum: {max(token_counts)}")
        print(f"Moyenne: {sum(token_counts) / len(token_counts):.2f}")
    except Exception as e:
        print(f"\nErreur lors du calcul des stats de tokens: {e}")

print("\nExploration du dataset terminée.")
