#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de démonstration et d'utilitaires pour le tokenizer multimodal NeuroLite.

Ce script permet de :
1. Créer et configurer un tokenizer multimodal
2. Traiter différentes modalités (texte, image, audio, vidéo, graphe)
3. Visualiser et analyser les tokens générés
4. Sauvegarder/charger des tokens et configurations
5. Intégrer le tokenizer avec le modèle NeuroLite

Exemples d'utilisation:
    python tokenizer.py --mode demo
    python tokenizer.py --mode analyze --input image.jpg
    python tokenizer.py --mode process --config custom_config.json --output tokens.pkl
"""

import os
import sys
import json
import argparse
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

import torch
import numpy as np
from PIL import Image

# Assurez-vous que le package neurolite est dans le PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurolite.tokenization import NeuroLiteTokenizer, TokenizerConfig
from neurolite.core.model import NeuroLiteModel
from neurolite.Configs.config import NeuroLiteConfig

# Configuration par défaut pour le tokenizer
DEFAULT_CONFIG = {
    "semantic_vocab_size": 8192,
    "detail_vocab_size": 32768,
    "hidden_size": 768,
    "dropout_rate": 0.1,
    "use_residual_vq": True,
    "num_residual_layers": 3,
    "commitment_weight": 0.25,
    "use_dual_codebook": True,
    "hierarchical_levels": 3,
    "use_context_modulation": True,
    "modality_alignment_weight": 0.5
}

# Chemins par défaut
DEFAULT_SAVE_DIR = Path(os.path.join(os.path.dirname(__file__), '../outputs/tokenizer'))
DEFAULT_SAVE_DIR.mkdir(parents=True, exist_ok=True)


def create_tokenizer(config_path: Optional[str] = None) -> NeuroLiteTokenizer:
    """
    Crée et configure un tokenizer multimodal.
    
    Args:
        config_path: Chemin optionnel vers un fichier de configuration JSON
        
    Returns:
        Tokenizer multimodal configuré
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
    else:
        config_dict = DEFAULT_CONFIG
    
    print(f"Création du tokenizer avec configuration: {config_dict}")
    config = TokenizerConfig(**config_dict)
    return NeuroLiteTokenizer(config)


def load_sample_inputs() -> Dict[str, Any]:
    """
    Charge des exemples d'entrées multimodales pour démonstration.
    
    Returns:
        Dictionnaire d'entrées pour différentes modalités
    """
    inputs = {
        'text': [
            "L'architecture universelle d'IA s'inspire des avancées récentes pour atteindre polyvalence et efficacité.",
            "Elle intègre des encodeurs modulaires spécialisés et un noyau latent universel partagé."
        ],
        'image': torch.randn(2, 3, 224, 224),  # Images simulées [batch, channels, height, width]
        'audio': torch.randn(2, 16000),        # Audio simulé [batch, time] - Format correct pour l'encodeur audio
        'video': torch.randn(2, 8, 3, 224, 224), # Vidéo simulée: [batch, frames, channels, height, width]
    }
    
    # Simuler des données de graphe
    batch_size = 2
    num_nodes = 16
    inputs['graph'] = {
        'node_features': torch.randn(batch_size, num_nodes, 64),
        'adjacency_matrix': torch.bernoulli(torch.ones(batch_size, num_nodes, num_nodes) * 0.3),
        'node_mask': torch.ones(batch_size, num_nodes)
    }
    
    return inputs


def process_and_tokenize(tokenizer: NeuroLiteTokenizer, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Traite et tokenize les entrées multimodales.
    
    Args:
        tokenizer: Tokenizer multimodal
        inputs: Dictionnaire d'entrées multimodales
        
    Returns:
        Dictionnaire des tokens et représentations générés
    """
    print("\nTraitement et tokenization des entrées multimodales...")
    tokens_output = tokenizer.tokenize(inputs)
    
    # Afficher les informations sur les tokens générés
    print("\nInformations sur les tokens générés:")
    for key, value in tokens_output.items():
        if isinstance(value, torch.Tensor):
            shape_info = f"Tensor de forme {list(value.shape)}"
            print(f"  - {key}: {shape_info}")
        elif isinstance(value, dict):
            print(f"  - {key}: Dictionnaire contenant {list(value.keys())}")
        else:
            print(f"  - {key}: {type(value)}")
    
    return tokens_output


def visualize_tokens(tokens_output: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Visualise les tokens générés.
    
    Args:
        tokens_output: Dictionnaire des tokens générés
        save_path: Chemin pour sauvegarder la visualisation
    """
    # Créer un répertoire pour sauvegarder plusieurs visualisations si nécessaire
    if save_path:
        base_dir = os.path.dirname(save_path)
        base_name = os.path.splitext(os.path.basename(save_path))[0]
    
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
    
    if save_path:
        plt.savefig(save_path)
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
            plt.hist(detail_indices, bins=50, alpha=0.7, color='darkorange')
            plt.title("Distribution des tokens détaillés")
            plt.xlabel("Indice de token")
            plt.ylabel("Fréquence")
            
            # Ajouter des statistiques
            plt.axvline(detail_indices.mean(), color='r', linestyle='dashed', linewidth=2, label=f'Moyenne: {detail_indices.mean():.2f}')
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            dist_path = os.path.join(base_dir, f"{base_name}_distribution.png")
            plt.savefig(dist_path)
            print(f"Distribution des tokens sauvegardée dans: {dist_path}")
    
    # Figure 3: Visualisation de l'espace latent (réduction de dimensionnalité ou PCA)
    if 'semantic_tokens' in tokens_output and isinstance(tokens_output['semantic_tokens'], torch.Tensor) and \
       'detail_tokens' in tokens_output and isinstance(tokens_output['detail_tokens'], torch.Tensor):
        
        plt.figure(figsize=(12, 10))
        
        # Préparation des données
        semantic_tokens = tokens_output['semantic_tokens'].detach().reshape(-1, tokens_output['semantic_tokens'].shape[-1]).cpu().numpy()
        detail_tokens = tokens_output['detail_tokens'].detach().reshape(-1, tokens_output['detail_tokens'].shape[-1]).cpu().numpy()
        
        # Vérifier si nous avons assez d'échantillons pour une visualisation significative
        min_samples_needed = 10  # Nombre minimum d'échantillons pour une visualisation significative
        
        if semantic_tokens.shape[0] >= min_samples_needed and detail_tokens.shape[0] >= min_samples_needed:
            # Utiliser t-SNE avec une perplexité adaptée
            from sklearn.manifold import TSNE
            
            # Calculer une perplexité adaptée (maximum 30, minimum 2, et toujours < nombre d'échantillons)
            perplexity = min(30, max(2, semantic_tokens.shape[0] // 3))
            
            # Réduction de dimensionnalité des tokens sémantiques
            tsne_semantic = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(semantic_tokens)
            
            plt.subplot(2, 1, 1)
            plt.scatter(tsne_semantic[:, 0], tsne_semantic[:, 1], c='steelblue', alpha=0.8)
            plt.title(f"Espace latent des tokens sémantiques (t-SNE, perplexité={perplexity})")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            
            # Réduction de dimensionnalité des tokens détaillés
            perplexity = min(30, max(2, detail_tokens.shape[0] // 3))
            tsne_detail = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(detail_tokens)
            
            plt.subplot(2, 1, 2)
            plt.scatter(tsne_detail[:, 0], tsne_detail[:, 1], c='darkorange', alpha=0.8)
            plt.title(f"Espace latent des tokens détaillés (t-SNE, perplexité={perplexity})")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
        else:
            # Fallback à PCA qui fonctionne même avec peu d'échantillons
            from sklearn.decomposition import PCA
            import matplotlib.patches as mpatches
            
            # Définir des palettes de couleurs pour représenter différentes modalités
            modality_colors = {'text': '#1f77b4', 'image': '#ff7f0e', 'audio': '#2ca02c',
                               'video': '#d62728', 'graph': '#9467bd'}
            marker_styles = {'text': 'o', 'image': 's', 'audio': '^', 'video': 'D', 'graph': 'X'}
            modality_names = {'text': 'Texte', 'image': 'Image', 'audio': 'Audio', 'video': 'Vidéo', 'graph': 'Graphe'}
            
            # Identifier les modalités des tokens en examinant encoded_features
            modalities = []
            if 'encoded_features' in tokens_output and isinstance(tokens_output['encoded_features'], dict):
                modalities = list(tokens_output['encoded_features'].keys())
            
            # Générer des modalités fictives si non disponibles
            if not modalities and semantic_tokens.shape[0] > 0:
                if semantic_tokens.shape[0] <= 5:
                    modalities = list(modality_colors.keys())[:semantic_tokens.shape[0]]
                else:
                    # Attribuer des modalités de manière cyclique pour les grands ensembles
                    modalities = [list(modality_colors.keys())[i % len(modality_colors)] for i in range(semantic_tokens.shape[0])]
            
            # Réduction de dimensionnalité des tokens sémantiques avec PCA
            if semantic_tokens.shape[0] > 1:  # Au moins 2 points pour PCA
                # Appliquer PCA et obtenir aussi la variance expliquée
                pca = PCA(n_components=2)
                pca_semantic = pca.fit_transform(semantic_tokens)
                explained_variance = pca.explained_variance_ratio_ * 100
                
                plt.figure(figsize=(12, 10))
                
                # Subplot pour l'espace sémantique
                ax1 = plt.subplot(2, 1, 1)
                
                # Visualisation améliorée avec modalités
                legend_handles = []
                for i, (x, y) in enumerate(pca_semantic):
                    modality = modalities[i % len(modalities)]
                    plt.scatter(x, y, c=modality_colors[modality], marker=marker_styles[modality], 
                                s=100, alpha=0.8, edgecolors='black', linewidths=0.5, 
                                label=modality_names[modality])
                    plt.annotate(f"{i+1}", (x, y), xytext=(5, 5), textcoords='offset points', 
                                 fontsize=9, fontweight='bold')
                    
                    # Préparer la légende
                    if modality_names[modality] not in [h.get_label() for h in legend_handles]:
                        legend_handles.append(plt.Line2D([0], [0], marker=marker_styles[modality], color='w', 
                                                    markerfacecolor=modality_colors[modality], markersize=10, 
                                                    label=modality_names[modality], markeredgecolor='black', 
                                                    markeredgewidth=0.5))
                
                # Ajouter légende et titre
                plt.legend(handles=legend_handles, loc='upper right', title="Modalités")
                plt.title(f"Espace latent des tokens sémantiques (PCA) - {semantic_tokens.shape[0]} échantillons")
                plt.xlabel(f"Composante principale 1 ({explained_variance[0]:.2f}% de variance)")
                plt.ylabel(f"Composante principale 2 ({explained_variance[1]:.2f}% de variance)")
                
                # Ajouter un peu de style
                plt.grid(True, linestyle='--', alpha=0.7)
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                
                # Calculer la distance moyenne entre points pour évaluer la séparation des modalités
                if semantic_tokens.shape[0] > 2:
                    from scipy.spatial.distance import pdist
                    distances = pdist(pca_semantic)
                    avg_distance = distances.mean()
                    plt.annotate(f"Distance moyenne: {avg_distance:.4f}", xy=(0.05, 0.05), 
                                 xycoords='axes fraction', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                                                                               fc="white", ec="gray", alpha=0.8))
                
            else:
                plt.subplot(2, 1, 1)
                plt.text(0.5, 0.5, "Pas assez d'échantillons pour la visualisation", 
                        ha='center', va='center', transform=plt.gca().transAxes)
            
            # Réduction de dimensionnalité des tokens détaillés avec PCA
            if detail_tokens.shape[0] > 1:  # Au moins 2 points pour PCA
                # Appliquer PCA et obtenir aussi la variance expliquée
                pca = PCA(n_components=2)
                pca_detail = pca.fit_transform(detail_tokens)
                explained_variance = pca.explained_variance_ratio_ * 100
                
                # Subplot pour l'espace détaillé
                ax2 = plt.subplot(2, 1, 2)
                
                # Visualisation améliorée avec modalités
                legend_handles = []
                for i, (x, y) in enumerate(pca_detail):
                    modality = modalities[i % len(modalities)]
                    plt.scatter(x, y, c=modality_colors[modality], marker=marker_styles[modality], 
                                s=100, alpha=0.8, edgecolors='black', linewidths=0.5, 
                                label=modality_names[modality])
                    plt.annotate(f"{i+1}", (x, y), xytext=(5, 5), textcoords='offset points', 
                                 fontsize=9, fontweight='bold')
                    
                    # Préparer la légende
                    if modality_names[modality] not in [h.get_label() for h in legend_handles]:
                        legend_handles.append(plt.Line2D([0], [0], marker=marker_styles[modality], color='w', 
                                                    markerfacecolor=modality_colors[modality], markersize=10, 
                                                    label=modality_names[modality], markeredgecolor='black', 
                                                    markeredgewidth=0.5))
                
                # Ajouter légende et titre
                plt.legend(handles=legend_handles, loc='upper right', title="Modalités")
                plt.title(f"Espace latent des tokens détaillés (PCA) - {detail_tokens.shape[0]} échantillons")
                plt.xlabel(f"Composante principale 1 ({explained_variance[0]:.2f}% de variance)")
                plt.ylabel(f"Composante principale 2 ({explained_variance[1]:.2f}% de variance)")
                
                # Ajouter un peu de style
                plt.grid(True, linestyle='--', alpha=0.7)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                
                # Calculer la distance moyenne entre points pour évaluer la séparation des modalités
                if detail_tokens.shape[0] > 2:
                    from scipy.spatial.distance import pdist
                    distances = pdist(pca_detail)
                    avg_distance = distances.mean()
                    plt.annotate(f"Distance moyenne: {avg_distance:.4f}", xy=(0.05, 0.05), 
                                 xycoords='axes fraction', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                                                                               fc="white", ec="gray", alpha=0.8))
            else:
                plt.subplot(2, 1, 2)
                plt.text(0.5, 0.5, "Pas assez d'échantillons pour la visualisation", 
                        ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        
        if save_path:
            latent_path = os.path.join(base_dir, f"{base_name}_latent_space.png")
            plt.savefig(latent_path)
            print(f"Visualisation de l'espace latent sauvegardée dans: {latent_path}")
        
    plt.close('all')  # Fermer toutes les figures pour éviter l'affichage en bloc
   

def save_tokens(tokens_output: Dict[str, Any], save_path: str) -> None:
    """
    Sauvegarde les tokens générés.
    
    Args:
        tokens_output: Dictionnaire des tokens générés
        save_path: Chemin pour sauvegarder les tokens
    """
    # Convertir les tensors en numpy pour la sauvegarde
    serializable_output = {}
    for key, value in tokens_output.items():
        if isinstance(value, torch.Tensor):
            serializable_output[key] = value.cpu().numpy()
        elif isinstance(value, dict):
            serializable_output[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, torch.Tensor):
                    serializable_output[key][subkey] = subvalue.cpu().numpy()
                else:
                    serializable_output[key][subkey] = subvalue
        else:
            serializable_output[key] = value
    
    with open(save_path, 'wb') as f:
        pickle.dump(serializable_output, f)
    
    print(f"Tokens sauvegardés dans {save_path}")


def integrate_with_model(tokenizer: NeuroLiteTokenizer, inputs: Dict[str, Any]) -> None:
    """
    Démontre l'intégration du tokenizer avec le modèle NeuroLite.
    
    Args:
        tokenizer: Tokenizer multimodal
        inputs: Dictionnaire d'entrées multimodales
    """
    print("\nIntégration avec le modèle NeuroLite...")
    
    # Créer une configuration pour le modèle
    model_config = NeuroLiteConfig(
        hidden_size=768,
        num_mixer_layers=6,
        use_multimodal_input=True,
        use_cross_modal_attention=True,
        multimodal_hidden_dim=768,
        multimodal_output_dim=768
    )
    
    # Initialiser le modèle
    model = NeuroLiteModel(
        config=model_config,
        task_type="multimodal_generation",
        tokenizer=tokenizer
    )
    
    print("Modèle initialisé avec tokenizer multimodal")
    
    # Exécuter une inférence
    print("Exécution d'une inférence de démonstration...")
    with torch.no_grad():
        outputs = model.generate(
            multimodal_inputs=inputs,
            target_modalities=["text", "image"],
            temperature=0.8
        )
    
    print("Inférence terminée. Types de sorties générées:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: Tensor de forme {list(value.shape)}")
        else:
            print(f"  - {key}: {type(value)}")


def analyze_tokens(tokens_output: Dict[str, Any]) -> None:
    """
    Analyse avancée des tokens générés.
    
    Args:
        tokens_output: Dictionnaire des tokens générés
    """
    print("\nAnalyse des tokens générés:")
    
    # Statistiques sur les tokens sémantiques
    if 'semantic_indices' in tokens_output and isinstance(tokens_output['semantic_indices'], torch.Tensor):
        semantic_indices = tokens_output['semantic_indices'].cpu().numpy()
        unique_semantic = np.unique(semantic_indices)
        
        print(f"Statistiques des tokens sémantiques:")
        print(f"  - Nombre total de tokens: {semantic_indices.size}")
        print(f"  - Nombre de tokens uniques: {len(unique_semantic)}")
        print(f"  - Diversité (unique/total): {len(unique_semantic)/semantic_indices.size:.4f}")
        
        # Histogramme des fréquences
        fig, ax = plt.subplots(figsize=(10, 6))
        counts, bins, _ = ax.hist(semantic_indices.flatten(), bins=50)
        ax.set_title("Distribution des tokens sémantiques")
        ax.set_xlabel("Indice de token")
        ax.set_ylabel("Fréquence")
        plt.show()
    
    # Analyse de la cohérence multimodale
    if all(k in tokens_output for k in ['text_tokens', 'image_tokens']) and \
       isinstance(tokens_output['text_tokens'], torch.Tensor) and \
       isinstance(tokens_output['image_tokens'], torch.Tensor):
        
        text_tokens = tokens_output['text_tokens']
        image_tokens = tokens_output['image_tokens']
        
        # Calculer la similarité cosinus entre les représentations
        text_norm = torch.norm(text_tokens, dim=-1, keepdim=True)
        image_norm = torch.norm(image_tokens, dim=-1, keepdim=True)
        similarity = torch.sum(text_tokens * image_tokens, dim=-1) / (text_norm * image_norm)
        
        print(f"\nSimilarité cosinus moyenne entre tokens texte et image: {similarity.mean().item():.4f}")


def main():
    """
    Fonction principale du script.
    """
    parser = argparse.ArgumentParser(description="Utilitaire pour le tokenizer multimodal NeuroLite")
    parser.add_argument("--mode", type=str, default="demo", 
                        choices=["demo", "analyze", "process", "visualize", "integrate"],
                        help="Mode d'exécution")
    parser.add_argument("--config", type=str, default=None, 
                        help="Chemin vers un fichier de configuration JSON")
    parser.add_argument("--input", type=str, default=None,
                        help="Chemin vers un fichier d'entrée ou répertoire d'entrées")
    parser.add_argument("--output", type=str, default=None,
                        help="Chemin pour la sortie")
    parser.add_argument("--save_visualization", action="store_true",
                        help="Sauvegarder la visualisation")
    
    args = parser.parse_args()
    
    # Créer le tokenizer
    tokenizer = create_tokenizer(args.config)
    
    # Générer un nom de fichier de sortie par défaut si non spécifié
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = str(DEFAULT_SAVE_DIR / f"tokens_{timestamp}.pkl")
    
    # Charger les entrées selon le mode
    if args.input and (args.mode == "process" or args.mode == "analyze"):
        # TODO: Implémenter le chargement d'entrées réelles à partir de fichiers
        inputs = load_sample_inputs()  # Pour l'instant, utilise des entrées simulées
    else:
        inputs = load_sample_inputs()
    
    # Traiter selon le mode
    if args.mode == "demo":
        print("\n=== Mode Démonstration du Tokenizer Multimodal NeuroLite ===\n")
        tokens_output = process_and_tokenize(tokenizer, inputs)
        
        # Générer un nom de fichier pour la visualisation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_path = str(DEFAULT_SAVE_DIR / f"tokenizer_visualization_{timestamp}.png")
        
        # Visualiser et sauvegarder automatiquement
        print(f"\nSauvegarde de la visualisation dans: {vis_path}")
        visualize_tokens(tokens_output, save_path=vis_path)
        analyze_tokens(tokens_output)
        
    elif args.mode == "analyze":
        print("\n=== Mode Analyse des Tokens ===\n")
        tokens_output = process_and_tokenize(tokenizer, inputs)
        analyze_tokens(tokens_output)
        
    elif args.mode == "process":
        print("\n=== Mode Traitement des Entrées ===\n")
        tokens_output = process_and_tokenize(tokenizer, inputs)
        save_tokens(tokens_output, args.output)
        
    elif args.mode == "visualize":
        print("\n=== Mode Visualisation des Tokens ===\n")
        tokens_output = process_and_tokenize(tokenizer, inputs)
        vis_path = args.output.replace(".pkl", ".png") if args.save_visualization else None
        visualize_tokens(tokens_output, save_path=vis_path)
        
    elif args.mode == "integrate":
        print("\n=== Mode Intégration avec Modèle ===\n")
        integrate_with_model(tokenizer, inputs)


if __name__ == "__main__":
    main()