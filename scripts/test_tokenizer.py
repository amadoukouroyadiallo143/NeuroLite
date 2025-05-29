#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de test pour le tokenizer multimodal NeuroLite entraîné.

Ce script permet de :
1. Charger le tokenizer multimodal entraîné
2. Générer des données de test
3. Tokenizer les données et visualiser les résultats
4. Analyser les performances du tokenizer

Exemple d'utilisation:
    python test_tokenizer.py --model_path ../outputs/tokenizer_training/checkpoints/final_model.pt
"""

import os
import sys
import json
import argparse
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import torch
import numpy as np
from tqdm import tqdm

# Assurez-vous que le package neurolite est dans le PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurolite.tokenization import NeuroLiteTokenizer, TokenizerConfig
from neurolite.tokenization.quantizers import ResidualVQ, DualCodebookVQ

# Chemins par défaut
DEFAULT_MODEL_PATH = Path(os.path.join(os.path.dirname(__file__), '../outputs/tokenizer_training/checkpoints/final_model.pt'))
DEFAULT_OUTPUT_DIR = Path(os.path.join(os.path.dirname(__file__), '../outputs/tokenizer_test'))
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_tokenizer(model_path: str) -> NeuroLiteTokenizer:
    """
    Charge un tokenizer entraîné à partir d'un fichier de checkpoint.
    
    Args:
        model_path: Chemin vers le fichier de checkpoint du tokenizer
        
    Returns:
        Tokenizer multimodal chargé
    """
    print(f"Chargement du tokenizer depuis: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier de modèle n'existe pas: {model_path}")
    
    # Importer les classes de configuration nécessaires
    from neurolite.tokenization.config import (
        TokenizerConfig,
        TextEncoderConfig,
        ImageEncoderConfig,
        AudioEncoderConfig,
        VideoEncoderConfig,
        GraphEncoderConfig
    )
    
    # Charger le checkpoint avec weights_only=False pour supporter les classes personnalisées
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=False  # Nécessaire pour charger les classes personnalisées
        )
        print("Checkpoint chargé avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement du checkpoint: {e}")
        raise
    
    # Vérifier si le checkpoint contient la configuration du tokenizer
    if 'tokenizer_config' not in checkpoint:
        raise ValueError("Le checkpoint ne contient pas de configuration de tokenizer (clé 'tokenizer_config' manquante)")
    
    # Récupérer la configuration du tokenizer
    tokenizer_config_dict = checkpoint['tokenizer_config']
    
    # Afficher les clés disponibles pour le débogage
    print("Clés disponibles dans le checkpoint:", list(checkpoint.keys()))
    
    # Convertir le dictionnaire de configuration en objet TokenizerConfig
    if isinstance(tokenizer_config_dict, dict):
        print("Conversion du dictionnaire de configuration en objet TokenizerConfig...")
        
        # Extraire les configurations spécifiques à chaque modalité
        text_config_dict = tokenizer_config_dict.pop('text_encoder_config', {})
        vision_config_dict = tokenizer_config_dict.pop('vision_encoder_config', {})
        audio_config_dict = tokenizer_config_dict.pop('audio_encoder_config', {})
        video_config_dict = tokenizer_config_dict.pop('video_encoder_config', {})
        graph_config_dict = tokenizer_config_dict.pop('graph_encoder_config', {})
        
        # Afficher les types pour le débogage
        print(f"Type de text_config_dict: {type(text_config_dict)}")
        if hasattr(text_config_dict, '__dict__'):
            print(f"Attributs de text_config_dict: {vars(text_config_dict).keys()}")
        
        # Créer les objets de configuration pour chaque modalité
        # Créer les objets de configuration pour chaque modalité
        text_config = text_config_dict if isinstance(text_config_dict, TextEncoderConfig) else TextEncoderConfig(**text_config_dict) if text_config_dict else TextEncoderConfig()
        vision_config = vision_config_dict if isinstance(vision_config_dict, ImageEncoderConfig) else ImageEncoderConfig(**vision_config_dict) if vision_config_dict else ImageEncoderConfig()
        audio_config = audio_config_dict if isinstance(audio_config_dict, AudioEncoderConfig) else AudioEncoderConfig(**audio_config_dict) if audio_config_dict else AudioEncoderConfig()
        video_config = video_config_dict if isinstance(video_config_dict, VideoEncoderConfig) else VideoEncoderConfig(**video_config_dict) if video_config_dict else VideoEncoderConfig()
        graph_config = graph_config_dict if isinstance(graph_config_dict, GraphEncoderConfig) else GraphEncoderConfig(**graph_config_dict) if graph_config_dict else GraphEncoderConfig()
        
        # Créer la configuration du tokenizer
        tokenizer_config = TokenizerConfig(**tokenizer_config_dict)
        
        # Définir les configurations spécifiques aux modalités
        tokenizer_config.text_encoder_config = text_config
        tokenizer_config.vision_encoder_config = vision_config
        tokenizer_config.audio_encoder_config = audio_config
        tokenizer_config.video_encoder_config = video_config
        tokenizer_config.graph_encoder_config = graph_config
        
        print("Configuration du tokenizer créée avec succès à partir du dictionnaire.")
    else:
        # Si la configuration est déjà un objet TokenizerConfig
        tokenizer_config = tokenizer_config_dict
    
    # Créer le tokenizer
    try:
        print("Création du tokenizer avec la configuration...")
        tokenizer = NeuroLiteTokenizer(tokenizer_config)
        print("Tokenizer créé avec succès")
    except Exception as e:
        print(f"Erreur lors de la création du tokenizer: {e}")
        raise
    
    # Charger les poids du modèle si disponibles
    if 'model_state_dict' in checkpoint:
        print("Chargement des poids du modèle...")
        try:
            tokenizer.load_state_dict(checkpoint['model_state_dict'])
            print("Poids du modèle chargés avec succès")
        except Exception as e:
            print(f"Avertissement: Erreur lors du chargement des poids: {e}")
            print("Tentative de chargement partiel...")
            tokenizer.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Chargement partiel réussi")
    
    return tokenizer


def generate_test_data() -> Dict[str, Any]:
    """
    Génère des données de test multimodales pour évaluer le tokenizer.
    
    Returns:
        Dictionnaire d'entrées pour différentes modalités
    """
    print("\nGénération de données de test multimodales...")
    
    # Simuler des données textuelles
    texts = [
        "L'architecture universelle d'IA s'inspire des avancées récentes pour atteindre polyvalence et efficacité.",
        "Elle intègre des encodeurs modulaires spécialisés et un noyau latent universel partagé.",
        "La mémoire neuronale multi-niveaux permet de traiter des séquences de plusieurs millions de tokens."
    ]
    
    # Simuler des données pour différentes modalités
    batch_size = len(texts)
    inputs = {
        'text': texts,
        'image': torch.randn(batch_size, 3, 224, 224),  # Images simulées [batch, channels, height, width]
        'audio': torch.randn(batch_size, 16000),        # Audio simulé [batch, time]
        'video': torch.randn(batch_size, 8, 3, 224, 224), # Vidéo simulée: [batch, frames, channels, height, width]
    }
    
    # Simuler des données de graphe
    num_nodes = 16
    inputs['graph'] = {
        'node_features': torch.randn(batch_size, num_nodes, 64),
        'adjacency_matrix': torch.bernoulli(torch.ones(batch_size, num_nodes, num_nodes) * 0.3),
        'node_mask': torch.ones(batch_size, num_nodes)
    }
    
    print(f"Données de test générées pour {batch_size} exemples avec modalités: text, image, audio, video, graph")
    return inputs


def process_and_visualize(tokenizer: NeuroLiteTokenizer, inputs: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Traite les données avec le tokenizer et visualise les résultats.
    
    Args:
        tokenizer: Tokenizer multimodal
        inputs: Dictionnaire d'entrées multimodales
        output_dir: Répertoire pour les sorties
        
    Returns:
        Dictionnaire des tokens et représentations générés
    """
    print("\nTraitement et tokenization des entrées multimodales...")
    
    # Passer en mode évaluation
    tokenizer.eval()
    
    # Tokenizer les entrées
    with torch.no_grad():
        tokens_output = tokenizer.tokenize(inputs)
    
    print("\nInformations sur les tokens générés:")
    for key, value in tokens_output.items():
        if isinstance(value, torch.Tensor):
            shape_info = f"Tensor de forme {list(value.shape)}"
            print(f"  - {key}: {shape_info}")
        elif isinstance(value, dict):
            print(f"  - {key}: Dictionnaire contenant {list(value.keys())}")
        else:
            print(f"  - {key}: {type(value)}")
    
    # Visualiser les tokens
    visualize_tokens(tokens_output, output_dir)
    
    return tokens_output


def visualize_tokens(tokens_output: Dict[str, Any], output_dir: Path) -> None:
    """
    Visualise les tokens générés.
    
    Args:
        tokens_output: Dictionnaire des tokens générés
        output_dir: Répertoire pour sauvegarder les visualisations
    """
    print("\nCréation des visualisations...")
    
    # Figure 1: Carte thermique des indices de tokens (sémantiques et détaillés)
    plt.figure(figsize=(12, 10))
    
    # Visualiser les tokens sémantiques
    if 'semantic_indices' in tokens_output and isinstance(tokens_output['semantic_indices'], torch.Tensor):
        semantic_indices = tokens_output['semantic_indices'].cpu().numpy()
        plt.subplot(2, 1, 1)
        plt.imshow(semantic_indices, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title("Indices des tokens sémantiques")
        plt.xlabel("Position")
        plt.ylabel("Exemple")
    
    # Visualiser les tokens détaillés
    if 'detail_indices' in tokens_output and isinstance(tokens_output['detail_indices'], torch.Tensor):
        detail_indices = tokens_output['detail_indices'].cpu().numpy()
        plt.subplot(2, 1, 2)
        plt.imshow(detail_indices, aspect='auto', cmap='plasma')
        plt.colorbar()
        plt.title("Indices des tokens détaillés")
        plt.xlabel("Position")
        plt.ylabel("Exemple")
    
    plt.tight_layout()
    save_path = output_dir / "token_heatmap.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Carte thermique des tokens sauvegardée dans: {save_path}")
    
    # Figure 2: Distribution des tokens (histogrammes)
    if ('semantic_indices' in tokens_output and isinstance(tokens_output['semantic_indices'], torch.Tensor)) or \
       ('detail_indices' in tokens_output and isinstance(tokens_output['detail_indices'], torch.Tensor)):
        
        plt.figure(figsize=(12, 8))
        
        # Histogramme des tokens sémantiques
        if 'semantic_indices' in tokens_output and isinstance(tokens_output['semantic_indices'], torch.Tensor):
            semantic_indices = tokens_output['semantic_indices'].cpu().numpy().flatten()
            plt.subplot(2, 1, 1)
            plt.hist(semantic_indices, bins=50, alpha=0.7, color='steelblue')
            plt.title("Distribution des tokens sémantiques")
            plt.xlabel("Indice de token")
            plt.ylabel("Fréquence")
            
            # Ajouter des statistiques
            plt.axvline(semantic_indices.mean(), color='r', linestyle='dashed', linewidth=2, label=f'Moyenne: {semantic_indices.mean():.2f}')
            plt.legend()
        
        # Histogramme des tokens détaillés
        if 'detail_indices' in tokens_output and isinstance(tokens_output['detail_indices'], torch.Tensor):
            detail_indices = tokens_output['detail_indices'].cpu().numpy().flatten()
            plt.subplot(2, 1, 2)
            plt.hist(detail_indices, bins=50, alpha=0.7, color='lightcoral')
            plt.title("Distribution des tokens détaillés")
            plt.xlabel("Indice de token")
            plt.ylabel("Fréquence")
            
            # Ajouter des statistiques
            plt.axvline(detail_indices.mean(), color='darkred', linestyle='dashed', linewidth=2, label=f'Moyenne: {detail_indices.mean():.2f}')
            plt.legend()
        
        plt.tight_layout()
        save_path = output_dir / "token_distribution.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Distribution des tokens sauvegardée dans: {save_path}")
    
    # Figure 3: Visualisation des embeddings latents si disponibles
    if 'latent_embeddings' in tokens_output and isinstance(tokens_output['latent_embeddings'], torch.Tensor):
        latent_embeddings = tokens_output['latent_embeddings'].cpu().numpy()
        
        if latent_embeddings.ndim > 2:
            # Réduire les dimensions pour visualisation
            latent_embeddings = latent_embeddings.reshape(-1, latent_embeddings.shape[-1])
        
        # Utiliser PCA pour réduire à 2D pour visualisation
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_embeddings)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=10)
        plt.title("Projection 2D des embeddings latents")
        plt.xlabel("Dimension principale 1")
        plt.ylabel("Dimension principale 2")
        plt.grid(alpha=0.3)
        
        save_path = output_dir / "latent_embeddings.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Projection des embeddings latents sauvegardée dans: {save_path}")


def analyze_tokens(tokens_output: Dict[str, Any], output_dir: Path) -> None:
    """
    Analyse les tokens générés et produit des statistiques.
    
    Args:
        tokens_output: Dictionnaire des tokens générés
        output_dir: Répertoire pour les sorties
    """
    print("\nAnalyse des tokens générés...")
    
    results = {}
    
    # Analyser les tokens sémantiques
    if 'semantic_indices' in tokens_output and isinstance(tokens_output['semantic_indices'], torch.Tensor):
        semantic_indices = tokens_output['semantic_indices'].cpu().numpy()
        unique_tokens = np.unique(semantic_indices)
        vocab_size = tokens_output.get('semantic_vocab_size', 8192)
        
        results['semantic_tokens'] = {
            'vocab_size': vocab_size,
            'unique_tokens_used': len(unique_tokens),
            'vocabulary_usage_percent': round(len(unique_tokens) / vocab_size * 100, 2),
            'most_common_tokens': np.bincount(semantic_indices.flatten()).argsort()[-5:].tolist()
        }
        
    # Analyser les tokens détaillés
    if 'detail_indices' in tokens_output and isinstance(tokens_output['detail_indices'], torch.Tensor):
        detail_indices = tokens_output['detail_indices'].cpu().numpy()
        unique_tokens = np.unique(detail_indices)
        vocab_size = tokens_output.get('detail_vocab_size', 32768)
        
        results['detail_tokens'] = {
            'vocab_size': vocab_size,
            'unique_tokens_used': len(unique_tokens),
            'vocabulary_usage_percent': round(len(unique_tokens) / vocab_size * 100, 2),
            'most_common_tokens': np.bincount(detail_indices.flatten()).argsort()[-5:].tolist()
        }
    
    # Analyser les similitudes cross-modales si disponibles
    if 'cross_modal_similarity' in tokens_output and isinstance(tokens_output['cross_modal_similarity'], torch.Tensor):
        cross_modal_sim = tokens_output['cross_modal_similarity'].cpu().numpy()
        
        results['cross_modal_similarity'] = {
            'mean': float(np.mean(cross_modal_sim)),
            'max': float(np.max(cross_modal_sim)),
            'min': float(np.min(cross_modal_sim))
        }
    
    # Sauvegarder les résultats
    results_path = output_dir / "token_analysis.json"
    try:
        with open(results_path, 'w', encoding='utf-8') as f:
            # Convertir les objets numpy en types Python standards
            clean_results = json.loads(json.dumps(results, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
            json.dump(clean_results, f, indent=2)
        print(f"Analyse des tokens sauvegardée dans: {results_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de l'analyse: {e}")
        print("Résultats de l'analyse:")
        print(results)


def main() -> None:
    """
    Fonction principale du script.
    """
    parser = argparse.ArgumentParser(description="Test du tokenizer multimodal NeuroLite")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH),
                      help="Chemin vers le fichier de modèle entraîné")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                      help="Répertoire pour les sorties")
    args = parser.parse_args()
    
    # Convertir en Path
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Déterminer le dispositif (GPU si disponible)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du dispositif: {device}")
    
    try:
        # Charger le tokenizer
        tokenizer = load_tokenizer(model_path)
        tokenizer.to(device)
        
        # Générer des données de test
        test_data = generate_test_data()
        
        # Traiter les données et visualiser les résultats
        tokens_output = process_and_visualize(tokenizer, test_data, output_dir)
        
        # Analyser les tokens
        analyze_tokens(tokens_output, output_dir)
        
        print(f"\nTest du tokenizer terminé. Résultats sauvegardés dans: {output_dir}")
        
    except Exception as e:
        print(f"Erreur lors du test du tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    main()
