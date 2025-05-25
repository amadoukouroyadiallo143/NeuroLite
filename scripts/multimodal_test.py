#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de test et démonstration pour les fonctionnalités multimodales de NeuroLite.

Ce script permet de tester l'architecture multimodale en visualisant:
1. Les encodages pour chaque modalité (texte, image, audio, vidéo, graphe)
2. Les interactions cross-modales et l'alignement des représentations
3. Le comportement du noyau latent universel
4. Les performances de génération multimodale

Exemples d'utilisation:
    python multimodal_test.py --mode visualize
    python multimodal_test.py --mode benchmark --modalities text,image,audio
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# Assurez-vous que le package neurolite est dans le PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurolite.tokenization import NeuroLiteTokenizer, TokenizerConfig
from neurolite.core.model import NeuroLiteModel, NeuroLiteConfig
from neurolite.multimodal.multimodal import MultimodalProjection

# Répertoire pour sauvegarder les résultats
RESULTS_DIR = Path(os.path.join(os.path.dirname(__file__), '../outputs/multimodal_tests'))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_test_data(modalities: List[str] = None) -> Dict[str, Any]:
    """
    Charge des données de test pour les modalités spécifiées.
    
    Args:
        modalities: Liste des modalités à charger (texte, image, audio, vidéo, graphe)
        
    Returns:
        Dictionnaire de données de test pour chaque modalité
    """
    all_modalities = ['text', 'image', 'audio', 'video', 'graph']
    modalities = modalities or all_modalities
    
    data = {}
    batch_size = 2
    
    # Filtrer les modalités valides
    valid_modalities = [m for m in modalities if m in all_modalities]
    
    for modality in valid_modalities:
        if modality == 'text':
            data[modality] = [
                "L'architecture universelle d'IA intègre des encodeurs modulaires spécialisés et un noyau latent partagé.",
                "Une mémoire neurale multi-niveaux permet au système de traiter des séquences extrêmes et de retenir des informations."
            ]
        elif modality == 'image':
            # Simuler des tenseurs d'image [B, C, H, W]
            data[modality] = torch.randn(batch_size, 3, 224, 224)
        elif modality == 'audio':
            # Simuler des tenseurs audio [B, T] - Format attendu par AudioEncoder
            sample_rate = 16000
            duration_sec = 5
            data[modality] = torch.randn(batch_size, sample_rate * duration_sec)
        elif modality == 'video':
            # Simuler des tenseurs vidéo [B, F, C, H, W] - avec taille d'image compatible (224x224)
            data[modality] = torch.randn(batch_size, 16, 3, 224, 224)
        elif modality == 'graph':
            # Simuler des données de graphe
            num_nodes = 24
            data[modality] = {
                'node_features': torch.randn(batch_size, num_nodes, 64),
                'adjacency_matrix': torch.bernoulli(torch.ones(batch_size, num_nodes, num_nodes) * 0.3),
                'node_mask': torch.ones(batch_size, num_nodes)
            }
    
    return data


def create_model(config_path: Optional[str] = None) -> Tuple[NeuroLiteModel, NeuroLiteTokenizer]:
    """
    Crée un modèle NeuroLite et un tokenizer pour les tests.
    
    Args:
        config_path: Chemin optionnel vers un fichier de configuration JSON
        
    Returns:
        Tuple de (modèle, tokenizer)
    """
    # Charger la configuration si spécifiée
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            model_config = config_dict.get('model_config', {})
            tokenizer_config = config_dict.get('tokenizer_config', {})
    else:
        # Configurations par défaut
        model_config = {
            'hidden_size': 768,
            'num_mixer_layers': 6,
            'use_multimodal_input': True,
            'use_cross_modal_attention': True,
            'multimodal_hidden_dim': 768,
            'multimodal_output_dim': 768,
            'use_hierarchical_memory': True,
            'memory_size': 1024,
            'memory_levels': 3
        }
        
        tokenizer_config = {
            'semantic_vocab_size': 8192,
            'detail_vocab_size': 32768,
            'hidden_size': 768,
            'dropout_rate': 0.1,
            'use_residual_vq': True,
            'use_dual_codebook': True
        }
    
    # Créer le tokenizer
    tokenizer_cfg = TokenizerConfig(**tokenizer_config)
    tokenizer = NeuroLiteTokenizer(tokenizer_cfg)
    
    # Créer le modèle
    model_cfg = NeuroLiteConfig(**model_config)
    model = NeuroLiteModel(
        config=model_cfg,
        task_type="multimodal_generation",
        tokenizer=tokenizer
    )
    
    return model, tokenizer


def visualize_modality_encodings(model: NeuroLiteModel, data: Dict[str, Any]) -> None:
    """
    Visualise les encodages pour chaque modalité.
    
    Args:
        model: Modèle NeuroLite
        data: Données d'entrée par modalité
    """
    print("\nAnalyse des encodages par modalité...")
    
    # Utiliser directement le module de projection multimodale
    projection = model.input_projection
    
    # Collecter les représentations individuelles
    with torch.no_grad():
        # Obtenir les représentations individuelles
        representations = projection(data, return_individual_modalities=True)
        
        # Le second élément contient les représentations par modalité
        if isinstance(representations, tuple) and len(representations) > 1:
            modality_encodings = representations[1]
            combined_encoding = representations[0]
        else:
            print("Erreur: La projection n'a pas retourné les représentations individuelles.")
            return
    
    # Visualiser les représentations par modalité
    plt.figure(figsize=(15, 10))
    
    # Extraire les noms des modalités disponibles
    modalities = list(modality_encodings.keys())
    num_modalities = len(modalities)
    
    # Calculer les valeurs moyennes pour la visualisation
    for i, modality in enumerate(modalities):
        encoding = modality_encodings[modality]
        
        if isinstance(encoding, torch.Tensor):
            # Calculer les statistiques
            mean_values = encoding.mean(dim=0).cpu().numpy()
            std_values = encoding.std(dim=0).cpu().numpy()
            
            # Tracer la représentation moyenne
            plt.subplot(num_modalities + 1, 1, i + 1)
            plt.plot(mean_values[:100])  # Visualiser les 100 premières dimensions
            plt.fill_between(
                range(100),
                mean_values[:100] - std_values[:100],
                mean_values[:100] + std_values[:100],
                alpha=0.3
            )
            plt.title(f"Encodage {modality}")
            plt.xlabel("Dimension")
            plt.ylabel("Valeur")
    
    # Visualiser la représentation combinée
    plt.subplot(num_modalities + 1, 1, num_modalities + 1)
    combined_mean = combined_encoding.mean(dim=0).cpu().numpy()
    combined_std = combined_encoding.std(dim=0).cpu().numpy()
    plt.plot(combined_mean[:100])
    plt.fill_between(
        range(100),
        combined_mean[:100] - combined_std[:100],
        combined_mean[:100] + combined_std[:100],
        alpha=0.3
    )
    plt.title("Encodage combiné")
    plt.xlabel("Dimension")
    plt.ylabel("Valeur")
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "modality_encodings.png")
    plt.show()
    
    # Analyser les similarités entre modalités
    similarity_matrix = np.zeros((num_modalities, num_modalities))
    modality_names = []
    
    encodings_list = []
    for modality in modalities:
        encoding = modality_encodings[modality]
        if isinstance(encoding, torch.Tensor):
            # Moyenner sur le batch
            mean_encoding = encoding.mean(dim=0).reshape(1, -1)
            encodings_list.append(mean_encoding)
            modality_names.append(modality)
    
    # Calculer la matrice de similarité
    for i in range(len(encodings_list)):
        for j in range(len(encodings_list)):
            enc_i = encodings_list[i]
            enc_j = encodings_list[j]
            
            # Normaliser
            enc_i_norm = F.normalize(enc_i, p=2, dim=1)
            enc_j_norm = F.normalize(enc_j, p=2, dim=1)
            
            # Similarité cosinus
            similarity = torch.mm(enc_i_norm, enc_j_norm.transpose(0, 1)).item()
            similarity_matrix[i, j] = similarity
    
    # Visualiser la matrice de similarité
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, vmin=-1, vmax=1, cmap='coolwarm')
    plt.colorbar(label="Similarité cosinus")
    plt.xticks(range(len(modality_names)), modality_names, rotation=45)
    plt.yticks(range(len(modality_names)), modality_names)
    plt.title("Similarité entre représentations des modalités")
    
    # Ajouter les valeurs dans les cellules
    for i in range(len(modality_names)):
        for j in range(len(modality_names)):
            plt.text(j, i, f"{similarity_matrix[i, j]:.2f}", 
                     ha="center", va="center", 
                     color="white" if abs(similarity_matrix[i, j]) > 0.5 else "black")
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "modality_similarity.png")
    plt.show()


def test_cross_modal_attention(model: NeuroLiteModel, data: Dict[str, Any]) -> None:
    """
    Teste et visualise l'attention cross-modale.
    
    Args:
        model: Modèle NeuroLite
        data: Données d'entrée par modalité
    """
    print("\nTest de l'attention cross-modale...")
    
    # Exécuter le modèle avec les entrées
    with torch.no_grad():
        outputs = model.forward(
            multimodal_inputs=data,
            output_hidden_states=True,
            return_dict=True
        )
    
    # Récupérer les états cachés
    if isinstance(outputs, dict) and 'hidden_states' in outputs:
        hidden_states = outputs['hidden_states']
        
        # Visualiser les états cachés
        if isinstance(hidden_states, torch.Tensor):
            # Réduire à 2D pour la visualisation
            states_2d = hidden_states[0].cpu().numpy()  # Premier exemple du batch
            
            plt.figure(figsize=(12, 8))
            plt.imshow(states_2d, aspect='auto', cmap='viridis')
            plt.colorbar(label="Valeur d'activation")
            plt.title("États cachés après traitement multimodal")
            plt.xlabel("Dimension")
            plt.ylabel("Position de séquence")
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / "hidden_states.png")
            plt.show()
    
    # Si le modèle expose les poids d'attention, les visualiser
    if hasattr(model.input_projection, 'cross_attention') and \
       hasattr(model.input_projection.cross_attention, 'last_attn_weights'):
        
        attn_weights = model.input_projection.cross_attention.last_attn_weights
        
        if isinstance(attn_weights, torch.Tensor):
            # Visualiser les poids d'attention
            attn_map = attn_weights[0].cpu().numpy()  # Premier exemple du batch
            
            plt.figure(figsize=(10, 8))
            plt.imshow(attn_map, cmap='hot', interpolation='nearest')
            plt.colorbar(label="Poids d'attention")
            plt.title("Carte d'attention cross-modale")
            plt.xlabel("Position source")
            plt.ylabel("Position cible")
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / "cross_modal_attention.png")
            plt.show()


def benchmark_performance(model: NeuroLiteModel, tokenizer: NeuroLiteTokenizer, 
                          modalities: List[str] = None) -> None:
    """
    Réalise un benchmark des performances du modèle.
    
    Args:
        model: Modèle NeuroLite
        tokenizer: Tokenizer NeuroLite
        modalities: Liste des modalités à tester
    """
    print("\nBenchmark des performances...")
    
    # Modalités disponibles
    all_modalities = ['text', 'image', 'audio', 'video', 'graph']
    test_modalities = modalities or all_modalities
    
    # Filtrer les modalités valides
    valid_modalities = [m for m in test_modalities if m in all_modalities]
    
    # Données pour les combinaisons de modalités
    combinations = []
    
    # Générer toutes les combinaisons de modalités
    for i in range(1, len(valid_modalities) + 1):
        for j in range(len(valid_modalities) - i + 1):
            combo = valid_modalities[j:j+i]
            combinations.append(combo)
    
    # Préparer les résultats
    results = {
        'combination': [],
        'tokenization_time': [],
        'forward_time': [],
        'generation_time': []
    }
    
    # Tester chaque combinaison
    for combo in combinations:
        print(f"\nTest de la combinaison: {', '.join(combo)}")
        
        # Charger les données pour cette combinaison
        data = load_test_data(combo)
        
        # Mesurer le temps de tokenization
        start_time = time.time()
        tokens = tokenizer.tokenize(data)
        tokenization_time = time.time() - start_time
        
        # Mesurer le temps de passage forward
        start_time = time.time()
        with torch.no_grad():
            _ = model.forward(multimodal_inputs=data, return_dict=True)
        forward_time = time.time() - start_time
        
        # Mesurer le temps de génération
        start_time = time.time()
        with torch.no_grad():
            _ = model.generate(
                multimodal_inputs=data,
                target_modalities=['text'],
                temperature=0.7
            )
        generation_time = time.time() - start_time
        
        # Enregistrer les résultats
        results['combination'].append('+'.join(combo))
        results['tokenization_time'].append(tokenization_time)
        results['forward_time'].append(forward_time)
        results['generation_time'].append(generation_time)
        
        print(f"  Temps de tokenization: {tokenization_time:.4f}s")
        print(f"  Temps de forward: {forward_time:.4f}s")
        print(f"  Temps de génération: {generation_time:.4f}s")
    
    # Visualiser les résultats
    plt.figure(figsize=(12, 8))
    
    # Barres pour les différentes mesures de temps
    bar_width = 0.25
    x = np.arange(len(results['combination']))
    
    # Tracer les barres
    plt.bar(x - bar_width, results['tokenization_time'], bar_width, label='Tokenization')
    plt.bar(x, results['forward_time'], bar_width, label='Forward Pass')
    plt.bar(x + bar_width, results['generation_time'], bar_width, label='Génération')
    
    plt.xlabel('Combinaison de modalités')
    plt.ylabel('Temps (secondes)')
    plt.title('Benchmark de performance par combinaison de modalités')
    plt.xticks(x, results['combination'], rotation=45)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(RESULTS_DIR / "performance_benchmark.png")
    plt.show()
    
    # Sauvegarder les résultats au format JSON
    with open(RESULTS_DIR / "benchmark_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


def test_latent_core(model: NeuroLiteModel, data: Dict[str, Any]) -> None:
    """
    Teste et visualise le comportement du noyau latent.
    
    Args:
        model: Modèle NeuroLite
        data: Données d'entrée par modalité
    """
    print("\nTest du noyau latent universel...")
    
    # Exécuter le modèle avec les entrées
    with torch.no_grad():
        outputs = model.forward(
            multimodal_inputs=data,
            output_hidden_states=True,
            return_dict=True
        )
    
    # Supposons que nous pouvons accéder aux latents du modèle
    # Cette partie dépend de l'implémentation spécifique du modèle
    if hasattr(model, 'latent_states') and model.latent_states is not None:
        latents = model.latent_states
        
        # Visualiser les latents
        if isinstance(latents, torch.Tensor):
            latents_np = latents.cpu().numpy()
            
            plt.figure(figsize=(12, 8))
            plt.imshow(latents_np[0], aspect='auto', cmap='plasma')  # Premier exemple du batch
            plt.colorbar(label="Valeur d'activation")
            plt.title("États latents du noyau universel")
            plt.xlabel("Dimension")
            plt.ylabel("Position latente")
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / "latent_core.png")
            plt.show()
    else:
        print("Les états latents ne sont pas accessibles directement dans le modèle actuel.")


def main():
    """
    Fonction principale du script.
    """
    parser = argparse.ArgumentParser(description="Tests multimodaux pour NeuroLite")
    parser.add_argument("--mode", type=str, default="visualize", 
                        choices=["visualize", "benchmark", "attention", "latent", "all"],
                        help="Mode de test")
    parser.add_argument("--config", type=str, default=None,
                        help="Chemin vers un fichier de configuration JSON")
    parser.add_argument("--modalities", type=str, default=None,
                        help="Liste de modalités à tester, séparées par des virgules")
    
    args = parser.parse_args()
    
    # Traiter les modalités si spécifiées
    modalities = None
    if args.modalities:
        modalities = args.modalities.split(',')
    
    # Créer le modèle et le tokenizer
    model, tokenizer = create_model(args.config)
    
    # Charger les données de test
    data = load_test_data(modalities)
    
    # Exécuter les tests selon le mode
    if args.mode == "visualize" or args.mode == "all":
        visualize_modality_encodings(model, data)
    
    if args.mode == "attention" or args.mode == "all":
        test_cross_modal_attention(model, data)
    
    if args.mode == "benchmark" or args.mode == "all":
        benchmark_performance(model, tokenizer, modalities)
    
    if args.mode == "latent" or args.mode == "all":
        test_latent_core(model, data)


if __name__ == "__main__":
    main()
