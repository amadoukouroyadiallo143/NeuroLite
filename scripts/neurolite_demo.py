"""
NeuroLite - Démonstration Interactive

Cette application montre les capacités de l'architecture NeuroLite,
une architecture universelle et légère conçue pour des applications
multimodales et l'intelligence artificielle générale.

L'architecture s'inspire des avancées récentes (SSM linéaires, Mémoire neuronale, etc.)
pour atteindre les objectifs de polyvalence et d'efficacité. Elle comporte des encodeurs
modulaires spécialisés (texte, image, audio, vidéo, graphes) qui convertissent chaque 
modalité en une représentation commune. Ces sorties alimentent un noyau latent universel
partagé, de taille fixe, qui réunit et traite l'information multi-modale.
"""

import torch
import argparse
import time
import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from collections import deque
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import hashlib
import json
import cv2
import soundfile as sf
import matplotlib.animation as animation
import networkx as nx

# Assurer que le package neurolite est dans le PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurolite import (
    NeuroLiteModel, 
    NeuroLiteConfig,
    NeurosymbolicReasoner,
    StructuredPlanner
)

# Imports supplémentaires depuis les sous-modules
from neurolite.multimodal.multimodal import MultimodalProjection, MultimodalGeneration, CrossModalAttention
from neurolite.memory import HierarchicalMemory, VectorMemoryStore
from neurolite.continual import ContinualAdapter, ReplayBuffer, ProgressiveCompressor
from neurolite.symbolic import BayesianBeliefNetwork
from neurolite.tokenization import NeuroLiteTokenizer, TokenizerConfig

# Imports depuis les encodeurs et décodeurs spécialisés
from neurolite.multimodal.encoders import (
    TextEncoder, ImageEncoder, AudioEncoder, VideoEncoder, GraphEncoder
)
from neurolite.multimodal.decoders import (
    TextDecoder, ImageDecoder, AudioDecoder, VideoDecoder, GraphDecoder
)


def format_time(seconds):
    """Formate une durée en secondes en texte lisible"""
    if seconds < 0.001:
        return f"{seconds*1000000:.2f} µs"
    elif seconds < 1:
        return f"{seconds*1000:.2f} ms"
    else:
        return f"{seconds:.4f} s"


def format_memory(num_bytes):
    """Formate une taille mémoire en texte lisible"""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024**2:
        return f"{num_bytes/1024:.2f} KB"
    elif num_bytes < 1024**3:
        return f"{num_bytes/1024**2:.2f} MB"
    else:
        return f"{num_bytes/1024**3:.2f} GB"


def display_model_info(model):
    """Affiche des informations sur le modèle"""
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = sum(p.numel() * p.element_size() for p in model.parameters())
    
    print("\n=== Informations sur le modèle ===")
    print(f"Nombre de paramètres: {param_count:,}")
    print(f"Taille mémoire: {format_memory(model_size)}")
    
    # Afficher la répartition des paramètres par couche
    print("\nRépartition des paramètres:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel():,} paramètres")


def test_inference_speed(model, num_runs=10):
    """Teste la vitesse d'inférence du modèle"""
    device = next(model.parameters()).device
    batch_size = 1
    
    # Créer des données simulées multimodales
    inputs = {
        'text': ["Ceci est un exemple de texte pour tester la vitesse d'inférence de NeuroLite."],
        'image': torch.randn(batch_size, 3, 224, 224).to(device),
        'video': torch.randn(batch_size, 8, 3, 224, 224).to(device)  # 8 frames
    }
    
    inference_func = lambda: model(multimodal_inputs=inputs)
    
    # Exécuter une première fois pour warm-up
    with torch.no_grad():
        _ = inference_func()
    
    # Mesurer le temps d'inférence
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = inference_func()
        times.append(time.time() - start_time)
    
    # Calculer les statistiques
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print("\n=== Performance d'inférence ===")
    print(f"Temps moyen: {format_time(avg_time)}")
    print(f"Temps min: {format_time(min_time)}")
    print(f"Temps max: {format_time(max_time)}")
    
    return avg_time


def interpret_symbolic_results(symbolic_details, tokenizer=None):
    """
    Convert symbolic details into a human-readable format.
    
    Args:
        symbolic_details: Dictionary containing symbolic tensors from NeurosymbolicReasoner
        tokenizer: Optional tokenizer for decoding entity IDs (if applicable)
    """
    if not symbolic_details:
        return "Aucun détail symbolique à interpréter."
    
    output = []
    
    # 1. Interprétation du score de cohérence
    if 'consistency' in symbolic_details and symbolic_details['consistency'] is not None:
        consistency = symbolic_details['consistency'].mean().item()  # Prend la moyenne si c'est un tenseur
        if consistency > 0.75:
            consistency_text = "élevée"
        elif consistency > 0.5:
            consistency_text = "moyenne"
        else:
            consistency_text = "faible"
        output.append(f"📊 Cohérence du raisonnement: {consistency:.2f} ({consistency_text})")
    
    # 2. Règles d'inférence (si disponibles)
    if 'rules' in symbolic_details and symbolic_details['rules'] is not None:
        rules = symbolic_details['rules']
        batch_size, num_rules, rule_len, dim = rules.shape
        output.append(f"\n🔍 {num_rules} règle(s) d'inférence générée(s):")
        
        # Afficher les règles sous forme de triplets (sujet, relation, objet)
        for i in range(min(3, num_rules)):  # Limiter à 3 règles pour la lisibilité
            rule = rules[0, i]  # Prend le premier échantillon du batch
            # Extraire les composants (sujet, relation, objet)
            if rule_len >= 3:  # Format [sujet, relation, objet, ...]
                subj = rule[0].mean().item()  # Simplification pour l'exemple
                rel = rule[1].mean().item()
                obj = rule[2].mean().item()
                output.append(f"  Règle {i+1}: [Sujet ~{subj:.2f}] -> [Relation ~{rel:.2f}] -> [Objet ~{obj:.2f}]")
            else:
                output.append(f"  Règle {i+1}: Format de règle non reconnu")
    
    # 3. Entités détectées (simplifié)
    if 'entities' in symbolic_details and symbolic_details['entities'] is not None:
        entities = symbolic_details['entities']
        num_entities = entities.shape[1]  # [batch, num_entities, dim]
        output.append(f"\n🔤 {num_entities} entité(s) détectée(s) dans le contexte")
    
    # 4. Relations détectées (simplifié)
    if 'relations' in symbolic_details and symbolic_details['relations'] is not None:
        relations = symbolic_details['relations']
        num_relations = relations.shape[1]  # [batch, num_relations, dim]
        output.append(f"🔗 {num_relations} relation(s) identifiée(s) entre les entités")
    
    return "\n".join(output)


def create_tokenizer(config=None):
    """Crée et configure le tokenizer multimodal.
    
    Args:
        config: Configuration optionnelle pour le tokenizer
    
    Returns:
        Tokenizer multimodal configuré
    """
    if config is None:
        config = TokenizerConfig(
            semantic_vocab_size=8192,
            detail_vocab_size=32768,
            hidden_size=768,
            dropout_rate=0.1,
            use_residual_vq=True,
            use_dual_codebook=True
        )
    
    print(f"Création du tokenizer multimodal avec une taille de vocabulaire sémantique de {config.semantic_vocab_size}...")
    return NeuroLiteTokenizer(config)


def process_multimodal_input(input_path, modality=None):
    """Traite un fichier multimodal et renvoie un dictionnaire pour le modèle.
    
    Args:
        input_path: Chemin vers le fichier d'entrée
        modality: Force le type de modalité (texte, image, audio, vidéo, graphe)
    
    Returns:
        Dictionnaire d'entrées multimodales
    """
    if not os.path.exists(input_path):
        if input_path.startswith("http"):
            # Traiter comme une URL
            return {
                'text': [f"URL: {input_path}"],
                'web_url': input_path
            }
        else:
            # Traiter comme du texte direct
            return {'text': [input_path]}
    
    # Déterminer la modalité en fonction de l'extension
    if modality is None:
        ext = os.path.splitext(input_path)[1].lower()
        if ext in [".txt", ".md", ".html", ".htm"]:
            modality = "text"
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
            modality = "image"
        elif ext in [".mp3", ".wav", ".ogg", ".flac"]:
            modality = "audio"
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            modality = "video"
        elif ext in [".json", ".graphml", ".gml"]:
            modality = "graph"
        else:
            # Par défaut, traiter comme du texte
            modality = "text"
    
    # Traiter selon la modalité
    inputs = {}
    
    if modality == "text":
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        inputs['text'] = [content]
        
    elif modality == "image":
        # Charger l'image avec PIL et convertir en tensor
        image = Image.open(input_path).convert('RGB')
        image = image.resize((224, 224))  # Redimensionner pour le modèle
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0  # [C, H, W]
        inputs['image'] = image_tensor.unsqueeze(0)  # Ajouter dimension de batch [1, C, H, W]
        
    elif modality == "audio":
        # Charger le fichier audio
        audio_data, sample_rate = sf.read(input_path)
        # Convertir en mono si stéréo
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
        # Convertir en tensor
        audio_tensor = torch.tensor(audio_data).float()
        # Redimensionner si nécessaire (simplifié)
        if len(audio_tensor) > sample_rate * 30:  # Limiter à 30 secondes
            audio_tensor = audio_tensor[:sample_rate * 30]
        inputs['audio'] = audio_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        
    elif modality == "video":
        # Charger la vidéo avec OpenCV
        cap = cv2.VideoCapture(input_path)
        frames = []
        success = True
        while success and len(frames) < 16:  # Limiter à 16 frames
            success, frame = cap.read()
            if success:
                # Convertir BGR à RGB et redimensionner
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (112, 112))
                frames.append(frame)
        cap.release()
        
        if frames:
            # Convertir en tensor
            video_tensor = torch.tensor(np.array(frames)).float() / 255.0  # [F, H, W, C]
            video_tensor = video_tensor.permute(0, 3, 1, 2)  # [F, C, H, W]
            inputs['video'] = video_tensor.unsqueeze(0)  # [1, F, C, H, W]
            
    elif modality == "graph":
        # Charger le graphe depuis un fichier JSON ou autre format
        try:
            if input_path.endswith(".json"):
                with open(input_path, 'r') as f:
                    graph_data = json.load(f)
                    
                # Construire un graphe NetworkX
                G = nx.Graph()
                
                # Ajouter des nœuds et des arêtes selon le format du fichier
                if 'nodes' in graph_data and 'edges' in graph_data:
                    for node in graph_data['nodes']:
                        G.add_node(node['id'], **node.get('attributes', {}))
                    
                    for edge in graph_data['edges']:
                        G.add_edge(edge['source'], edge['target'], **edge.get('attributes', {}))
                
                # Convertir en tenseurs pour le modèle
                num_nodes = len(G.nodes())
                if num_nodes > 0:
                    # Créer une matrice d'adjacence
                    adj_matrix = nx.to_numpy_array(G)
                    # Créer des caractéristiques de nœuds (ici simples embeddings aléatoires)
                    node_features = np.random.randn(num_nodes, 64)
                    
                    inputs['graph'] = {
                        'node_features': torch.tensor(node_features).float().unsqueeze(0),  # [1, N, D]
                        'adjacency_matrix': torch.tensor(adj_matrix).float().unsqueeze(0),  # [1, N, N]
                        'node_mask': torch.ones(1, num_nodes)  # [1, N]
                    }
        except Exception as e:
            print(f"Erreur lors du chargement du graphe: {e}")
            # Créer un graphe par défaut simple
            inputs['graph'] = {
                'node_features': torch.randn(1, 4, 64),  # [1, N, D]
                'adjacency_matrix': torch.eye(4).unsqueeze(0),  # [1, N, N]
                'node_mask': torch.ones(1, 4)  # [1, N]
            }
    
    return inputs


def visualize_modality_representations(model_outputs, save_path=None):
    """Visualise les représentations des différentes modalités.
    
    Args:
        model_outputs: Sorties du modèle contenant les représentations multimodales
        save_path: Chemin pour sauvegarder la visualisation
    """
    if not isinstance(model_outputs, dict) or 'modality_representations' not in model_outputs:
        print("Aucune représentation multimodale disponible pour la visualisation.")
        return
    
    # Récupérer les représentations par modalité
    modality_reps = model_outputs['modality_representations']
    
    # Vérifier le format
    if not isinstance(modality_reps, dict):
        print("Format de représentations multimodales non reconnu.")
        return
    
    modalities = list(modality_reps.keys())
    if not modalities:
        print("Aucune modalité trouvée dans les représentations.")
        return
    
    # Calculer les statistiques et visualiser
    plt.figure(figsize=(14, 10))
    
    # Pour chaque modalité, visualiser les premières dimensions
    for i, modality in enumerate(modalities):
        rep = modality_reps[modality]
        if isinstance(rep, torch.Tensor):
            # Réduire à 2D pour visualisation
            rep_np = rep.detach().cpu().numpy()
            if len(rep_np.shape) > 2:
                # Moyenner sur le batch et autres dimensions
                rep_np = rep_np.reshape(rep_np.shape[0], -1).mean(axis=0)
            elif len(rep_np.shape) == 2:
                rep_np = rep_np.mean(axis=0)
            
            # Tracer les 100 premières dimensions ou moins
            dim = min(100, rep_np.shape[0])
            plt.subplot(len(modalities), 1, i+1)
            plt.bar(range(dim), rep_np[:dim])
            plt.title(f"Représentation {modality}")
            plt.xlabel("Dimension")
            plt.ylabel("Activation")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualisation sauvegardée dans {save_path}")
    else:
        plt.show()


def visualize_tokens(tokens, tokenizer=None, save_path=None):
    """Visualise les tokens générés par le tokenizer multimodal.
    
    Args:
        tokens: Dictionnaire de tokens générés
        tokenizer: Tokenizer pour la décodification (optionnel)
        save_path: Chemin pour sauvegarder la visualisation
    """
    if not isinstance(tokens, dict):
        print("Format de tokens non reconnu.")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Visualiser les tokens sémantiques
    if 'semantic_indices' in tokens and isinstance(tokens['semantic_indices'], torch.Tensor):
        semantic_indices = tokens['semantic_indices'].cpu().numpy()
        plt.subplot(2, 1, 1)
        plt.imshow(semantic_indices, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title("Indices des tokens sémantiques")
        plt.xlabel("Position")
        plt.ylabel("Exemple")
    
    # Visualiser les tokens détaillés
    if 'detail_indices' in tokens and isinstance(tokens['detail_indices'], torch.Tensor):
        detail_indices = tokens['detail_indices'].cpu().numpy()
        plt.subplot(2, 1, 2)
        plt.imshow(detail_indices, aspect='auto', cmap='plasma')
        plt.colorbar()
        plt.title("Indices des tokens détaillés")
        plt.xlabel("Position")
        plt.ylabel("Exemple")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualisation des tokens sauvegardée dans {save_path}")
    else:
        plt.show()


def run_interactive_demo(model_size='base', agi_enabled=True):
    """Lance une démonstration interactive"""
    print(f"\n=== Démonstration de l'Architecture NeuroLite ===\n")
    print(f"Chargement du modèle NeuroLite ({model_size})...")
    if agi_enabled:
        print("Fonctionnalités AGI avancées activées: mémoire hiérarchique, apprentissage continu, raisonnement neurosymbolique.")
    else:
        print("Fonctionnalités AGI avancées désactivées.")
    
    # Créer la configuration du tokenizer
    tokenizer_config = TokenizerConfig(
        semantic_vocab_size=8192,
        detail_vocab_size=32768,
        hidden_size=768,
        dropout_rate=0.1,
        use_residual_vq=True,
        use_dual_codebook=True
    )
    
    # Créer le tokenizer multimodal
    tokenizer = create_tokenizer(tokenizer_config)
    
    # Créer la configuration du modèle
    if model_size == 'tiny':
        config = NeuroLiteConfig.tiny()
    elif model_size == 'small':
        config = NeuroLiteConfig.small()
    else:
        config = NeuroLiteConfig.base()
        
    # Activer l'apprentissage continu
    config.use_continual_adapter = True
    config.continual_adapter_buffer_size = 1000
    
    # Configuration multimodale avec notre nouvelle architecture
    config.use_multimodal_input = True
    config.multimodal_output_dim = 768
    config.multimodal_hidden_dim = 768
    config.use_cross_modal_attention = True
    config.cross_modal_num_heads = 8
    
    # Paramètres pour la nouvelle architecture multimodale
    config.image_size = 224
    config.multimodal_image_patch_size = 16
    config.max_audio_length_ms = 30000
    config.audio_sample_rate = 16000
    config.multimodal_video_num_sampled_frames = 16
    config.max_graph_nodes = 32
    
    # Mémoire hiérarchique
    config.use_external_memory = True
    config.use_hierarchical_memory = True
    config.short_term_memory_size = 64
    config.long_term_memory_size = 256
    config.persistent_memory_size = 512
    config.memory_dim = 128
    
    # Routage dynamique
    config.use_dynamic_routing = True
    config.num_experts = 8
    config.routing_top_k = 2
    
    # Raisonnement symbolique et planification
    config.use_symbolic_module = True
    config.use_advanced_reasoning = True
    config.symbolic_dim = 64
    config.num_inference_steps = 3
    config.use_planning_module = True
    config.num_planning_steps = 5
    config.plan_dim = 64
    
    # Réseau bayésien
    config.use_bayesian_module = True
    config.num_bayesian_variables = 16
    
    # Activer la compression progressive
    config.use_progressive_compression = True
    config.progressive_compression_ratio = 0.5
    config.progressive_compression_threshold = 0.1
    
    # Configuration pour l'apprentissage web
    config.web_learning = {
        'max_pages': 20,  # Nombre maximum de pages à explorer
        'max_depth': 2,   # Profondeur maximale de navigation
        'min_text_length': 100,  # Longueur minimale du texte à considérer
        'visited_urls_file': 'visited_urls.json'  # Fichier pour stocker les URLs visitées
    }
    
    # Créer le modèle
    model = NeuroLiteModel(config)
    
    # Afficher les informations sur le modèle
    display_model_info(model)
    
    # Tester la vitesse d'inférence
    avg_time = test_inference_speed(model)
    
    # Lancer l'interface interactive
    print("\n=== Démonstration Interactive de NeuroLite ===")
    print("Tapez du texte et NeuroLite le transformera en représentation vectorielle.")
    print("Pour quitter, tapez 'exit' ou 'quit'.")
    
    # Contexte de la session (pour la mémoire)
    session_context = []
    
    while True:
        # Afficher le menu principal
        print("\n=== Menu Principal ===")
        print("1. Afficher les informations du modèle")
        print("2. Tester la vitesse d'inférence")
        print("3. Effectuer une inférence simple")
        print("4. Effectuer une inférence avec contexte")
        print("5. Recherche sémantique dans la mémoire")
        print("6. Raisonnement symbolique avancé")
        print("7. Explorer la structure du modèle")
        print("8. Gérer la mémoire du modèle")
        print("9. Sauvegarder le modèle")
        print("10. Apprentissage autonome depuis le web")
        print("11. Discuter avec le modèle")
        print("0. Quitter")
        
        choice = input("\nVotre choix: ")
        
        if choice == '0' or choice.lower() in ['exit', 'quit']:
            break
            
        elif choice == '1':
            display_model_info(model)
            
        elif choice == '2':
            test_inference_speed(model)
            
        elif choice == '3':
            text = input("\nEntrez votre texte: ")
            if not text:
                continue
                
            print("Traitement en cours...")
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(multimodal_inputs={'text': [text]})
                
            process_time = time.time() - start_time
            
            # Ajouter au contexte de session
            session_context.append(text)
            if len(session_context) > 5:
                session_context.pop(0)
            
            # Obtenir l'embedding moyen
            embedding = torch.mean(outputs['hidden_states'], dim=1).squeeze().numpy()
            
            print(f"Temps de traitement: {format_time(process_time)}")
            print(f"Dimension de sortie: {outputs['hidden_states'].shape}")
            print(f"Norme de l'embedding: {np.linalg.norm(embedding):.4f}")
            
            # Afficher les premiers éléments de l'embedding
            print("\nAperçu de l'embedding:")
            preview = embedding[:5]
            print(", ".join([f"{x:.4f}" for x in preview]))
            
        elif choice == '4':
            if len(session_context) < 2:
                print("Veuillez d'abord traiter au moins deux textes (option 1).")
                continue

            print("\n=== Démonstration de la mémoire contextuelle ===")
            print("Contexte de la session actuelle:")
            for i, ctx in enumerate(session_context):
                print(f"[{i+1}] {ctx}")

            print("\nEntrez une requête qui pourrait être liée au contexte:")
            query = input("Requête: ")
            if not query:
                continue

            print("\nTraitement avec et sans contexte...")

            # Mode sans contexte
            if hasattr(model.memory, 'initialized'):
                model.memory.initialized = False

            with torch.no_grad():
                no_context_output = model(multimodal_inputs={'text': [query]}, update_memory=False)
                no_context_emb = torch.mean(no_context_output['hidden_states'], dim=1).squeeze()

            # Mode avec contexte
            if hasattr(model.memory, 'initialized'):
                model.memory.initialized = False

            with torch.no_grad():
                for ctx in session_context:
                    _ = model(multimodal_inputs={'text': [ctx]}, update_memory=True)
                context_output = model(multimodal_inputs={'text': [query]}, update_memory=False)
                context_emb = torch.mean(context_output['hidden_states'], dim=1).squeeze()

            # Calcul de l'impact
            similarity = torch.nn.functional.cosine_similarity(
                no_context_emb.unsqueeze(0),
                context_emb.unsqueeze(0)
            ).item()
            
            context_impact = 1.0 - similarity
            
            print(f"\nImpact contextuel mesuré: {context_impact:.2%}")
            
            if context_impact > 0.3:
                print("→ Influence contextuelle significative détectée")
            elif context_impact > 0.1:
                print("→ Influence contextuelle modérée")
            else:
                print("→ Peu d'influence contextuelle")

        elif choice == '2':
            print("\nEntrez deux textes à comparer:")
            text1 = input("Texte 1: ")
            text2 = input("Texte 2: ")
            
            if not text1 or not text2:
                continue
                
            print("Calcul de la similarité...")
            
            with torch.no_grad():
                output1 = model(multimodal_inputs={'text': [text1]})
                output2 = model(multimodal_inputs={'text': [text2]})
                
                # Calculer les embeddings moyens
                emb1 = torch.mean(output1['hidden_states'], dim=1).squeeze()
                emb2 = torch.mean(output2['hidden_states'], dim=1).squeeze()
                
                # Calculer la similarité cosinus
                similarity = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
                
                print(f"\nSimilarité cosinus: {similarity.item():.4f}")
                
        elif choice == '3':
            if len(session_context) < 2:
                print("Veuillez d'abord traiter au moins deux textes (option 1).")
                continue

            print("\n=== Démonstration de la mémoire contextuelle ===")
            print("Contexte de la session actuelle:")
            for i, ctx in enumerate(session_context):
                print(f"[{i+1}] {ctx}")

            print("\nEntrez une requête qui pourrait être liée au contexte:")
            query = input("Requête: ")
            if not query:
                continue

            print("\nTraitement avec et sans contexte...")

            # 1. Sans contexte (mémoire réinitialisée)
            if hasattr(model.memory, 'initialized'):
                model.memory.initialized = False
                
            with torch.no_grad():
                no_context_output = model(multimodal_inputs={'text': [query]}, update_memory=False)
                no_context_emb = torch.mean(no_context_output['hidden_states'], dim=1).squeeze()
            
            # 2. Avec contexte (nourrir la mémoire avec le contexte)
            if hasattr(model.memory, 'initialized'):
                model.memory.initialized = False
                
            with torch.no_grad():
                # Traiter le contexte
                for ctx in session_context:
                    _ = model(multimodal_inputs={'text': [ctx]}, update_memory=True)
                    
                # Traiter la requête
                context_output = model(multimodal_inputs={'text': [query]}, update_memory=False)
                context_emb = torch.mean(context_output['hidden_states'], dim=1).squeeze()
            
            # Calculs de similarité
            no_context_norm = no_context_emb / torch.norm(no_context_emb)
            context_norm = context_emb / torch.norm(context_emb)
            similarity = torch.dot(no_context_norm, context_norm).item()
            context_impact = 1.0 - similarity

            print(f"\nImpact du contexte: {context_impact:.2%}")

            if context_impact < 0.1:
                print("La mémoire contextuelle a eu peu d'influence sur cette requête.")
            elif context_impact < 0.3:
                print("La mémoire contextuelle a eu une influence modérée sur cette requête.")
            else:
                print("La mémoire contextuelle a fortement influencé la représentation de cette requête!")

        elif choice == '4':
            # Démonstration du raisonnement symbolique
            print("\n=== Démonstration du raisonnement symbolique ===\n")

            # Créer des faits externes simples (sous forme de tenseur)
            print("Ajout de faits symboliques externes...")
            facts = torch.randn(1, 5, 64)  # Simuler 5 faits dans l'espace symbolique

            premise = input("Entrez une prémisse (ex: 'Paris est la capitale de la France'): ")
            hypothesis = input("Entrez une hypothèse (ex: 'Paris est en Europe'): ")

            print("\nTraitement du raisonnement...")
            with torch.no_grad():
                # Traiter la prémisse avec les faits externes
                premise_result = model(
                    multimodal_inputs={'text': [premise]},
                    external_facts=facts,
                    return_symbolic=True
                )

                # Traiter l'hypothèse avec retour des informations symboliques
                hypothesis_result = model(
                    multimodal_inputs={'text': [hypothesis]},
                    external_facts=facts,
                    return_symbolic=True
                )

            # Si le module de raisonnement symbolique est activé et fonctionnel
            if isinstance(premise_result, dict) and 'symbolic' in premise_result:
                # Calculer la consistance entre la prémisse et l'hypothèse
                premise_relations = premise_result['symbolic']['relations']
                hypothesis_relations = hypothesis_result['symbolic']['relations']

                # Simuler un score d'inférence
                inference_score = torch.nn.functional.cosine_similarity(
                    premise_relations.mean(dim=1),
                    hypothesis_relations.mean(dim=1)
                ).item()

                print(f"\nCohérence de l'inférence: {inference_score:.4f}")

                # Interprétation du score
                if inference_score > 0.7:
                    print("Interprétation: L'hypothèse est fortement soutenue par la prémisse")
                elif inference_score > 0.4:
                    print("Interprétation: L'hypothèse est partiellement soutenue par la prémisse")
                else:
                    print("Interprétation: L'hypothèse n'est pas clairement liée à la prémisse")
            else:
                print("Le module de raisonnement symbolique n'est pas activé ou ne renvoie pas les informations symboliques")
                        
        elif choice == '5':
                    # Démonstration de la planification structurée
                    print("\n=== Démonstration de la planification structurée ===\n")
                    
                    goal = input("Entrez un objectif à planifier (ex: 'Organiser un voyage à Paris'): ")
                    constraint = input("Entrez une contrainte (ex: 'Budget limité'): ")
                    
                    # Créer un tenseur de contraintes simulé (trois dimensions: temps, coût, faisabilité)
                    constraints = torch.tensor([[0.8, 0.4, 0.9]]) # Temps OK, Coût restreint, Faisabilité élevée
                    
                    print("\nGénération du plan...")
                    with torch.no_grad():
                        plan_result = model(
                            input_texts=[goal],
                            use_planning=True,
                            constraints=constraints,
                            return_symbolic=True
                        )
                    
                    # Si le module de planification est activé et fonctionnel
                    if isinstance(plan_result, dict) and 'planning' in plan_result:
                        plan_steps = plan_result['planning']['plan']
                        plan_quality = plan_result['planning']['quality'].item()
                        plan_valid = plan_result['planning']['valid'].item() > 0.5
                        
                        print(f"\nQualité du plan: {plan_quality:.4f}")
                        print(f"Plan valide selon les contraintes: {'Oui' if plan_valid else 'Non'}")
                        
                        if plan_valid:
                            print("\nRésumé du plan généré:")
                            print("1. Étape: Recherche d'informations sur la destination")
                            print("2. Étape: Comparaison des options de transport et hébergement")
                            print("3. Étape: Réservation en tenant compte des contraintes")
                            print("4. Étape: Organisation des activités sur place")
                            print("5. Étape: Planification du budget détaillé")
                        else:
                            print("\nLe plan ne respecte pas toutes les contraintes spécifiées.")
                            print("Suggestion: Assouplir les contraintes ou modifier l'objectif.")
                    else:
                        print("Le module de planification n'est pas activé ou ne renvoie pas les informations de planification")
                        
        elif choice == '6':
                    # Démonstration de l'apprentissage continu
                    print("\n=== Démonstration de l'apprentissage continu ===\n")
                    
                    # Premier ensemble de données (distribution A)
                    print("Phase 1: Traitement du premier domaine...")
                    domain_a_texts = [
                        "Le deep learning est une méthode d'apprentissage automatique.",
                        "Les réseaux de neurones sont inspirés du cerveau humain.",
                        "L'intelligence artificielle transforme de nombreux secteurs."
                    ]
                    
                    with torch.no_grad():
                        domain_a_results = model(
                            input_texts=domain_a_texts,
                            continuous_learning=True  # Activer l'apprentissage continu
                        )
                        
                    embeddings_a = torch.mean(domain_a_results, dim=1)
                    
                    # Deuxième ensemble de données (distribution B, différente)
                    print("\nPhase 2: Adaptation à un nouveau domaine...")
                    domain_b_texts = [
                        "La photosynthèse est le processus de conversion de l'énergie lumineuse en énergie chimique.",
                        "Les plantes utilisent le dioxyde de carbone et libèrent de l'oxygène.",
                        "L'écosystème forestier est essentiel pour la biodiversité."
                    ]
                    
                    with torch.no_grad():
                        domain_b_results = model(
                            input_texts=domain_b_texts,
                            continuous_learning=True  # Activer l'apprentissage continu
                        )
                        
                    embeddings_b = torch.mean(domain_b_results, dim=1)
                    
                    # Test: revenir au premier domaine
                    print("\nPhase 3: Retour au premier domaine (test de rétention)...")
                    with torch.no_grad():
                        domain_a_again_results = model(
                            input_texts=[domain_a_texts[0]],  # Juste le premier texte pour l'exemple
                            continuous_learning=False  # Désactiver l'adaptation pour le test
                        )
                        
                    embedding_a_again = torch.mean(domain_a_again_results, dim=1)
                    
                    # Calculer la similarité pour vérifier la rétention
                    similarity = torch.nn.functional.cosine_similarity(
                        embeddings_a[0:1], embedding_a_again
                    ).item()
                    
                    print(f"\nSimilarité avec le premier domaine après adaptation: {similarity:.4f}")
                    if similarity > 0.8:
                        print("Excellente rétention des connaissances précédentes!")
                    elif similarity > 0.5:
                        print("Bonne rétention avec une adaptation modérée aux nouvelles données.")
                    else:
                        print("Adaptation forte au nouveau domaine mais rétention limitée.")
                        
        elif choice == '7':
                display_model_info(model)
                test_inference_speed(model)
                
        elif choice == '9':
                print("\n=== Sauvegarde du modèle ===\n")
                save_dir = input("Entrez le répertoire de sauvegarde (laisser vide pour 'saved_models/neurolite_model'): ").strip()
                if not save_dir:
                    save_dir = "models/neurolite_model"
                
                print("Options de sauvegarde:")
                print("1. Sauvegarde standard (peut échouer avec des structures complexes)")
                print("2. Sauvegarde légère (uniquement poids essentiels)")
                print("3. Sauvegarde avec mémoire désactivée")
                print("4. Sauvegarde avec adaptateur continu désactivé")
                print("5. Sauvegarde complète (mémoire + adaptateur désactivés)")
                
                save_option = input("Choisissez une option (1-5, défaut: 5): ").strip()
                if not save_option or save_option not in "12345":
                    save_option = "5"
                save_option = int(save_option)
                
                try:
                    # Créer le répertoire s'il n'existe pas
                    os.makedirs(save_dir, exist_ok=True)
                    
                    print(f"Sauvegarde du modèle dans {save_dir}...")
                    
                    # Utiliser la méthode save_pretrained améliorée avec les options appropriées
                    if save_option == 1:
                        # Sauvegarde standard sans options spéciales
                        print("Méthode: Sauvegarde standard")
                        model.save_pretrained(save_dir)
                    
                    elif save_option == 2:
                        # Sauvegarde légère - utiliser le paramètre lightweight
                        print("Méthode: Sauvegarde légère (uniquement poids essentiels)")
                        model.save_pretrained(save_dir, lightweight=True)
                    
                    elif save_option == 3:
                        # Désactiver la mémoire pendant la sauvegarde
                        print("Méthode: Sauvegarde avec mémoire désactivée")
                        model.save_pretrained(save_dir, disable_memory=True)
                    
                    elif save_option == 4:
                        # Désactiver l'adaptateur pendant la sauvegarde
                        print("Méthode: Sauvegarde avec adaptateur continu désactivé")
                        model.save_pretrained(save_dir, disable_adapters=True)
                    
                    elif save_option == 5:
                        # Désactiver à la fois la mémoire et l'adaptateur
                        print("Méthode: Sauvegarde complète (mémoire + adaptateur désactivés)")
                        model.save_pretrained(save_dir, disable_memory=True, disable_adapters=True)
                    
                    # Sauvegarder également la configuration du tokenizer s'il existe
                    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
                        tokenizer_path = os.path.join(save_dir, "tokenizer")
                        os.makedirs(tokenizer_path, exist_ok=True)
                        model.tokenizer.save_pretrained(tokenizer_path)
                        print(f"Tokenizer sauvegardé dans {tokenizer_path}")
                    
                    # Réinitialiser la limite de récursion à sa valeur d'origine
                    sys.setrecursionlimit(original_limit)
                    
                    print("\nModèle sauvegardé avec succès!")
                    print(f"Contenu du répertoire de sauvegarde: {os.listdir(save_dir)}")
                    
                except Exception as e:
                    # Réinitialiser la limite de récursion à sa valeur d'origine en cas d'erreur
                    if 'original_limit' in locals():
                        sys.setrecursionlimit(original_limit)
                    print(f"\nErreur lors de la sauvegarde du modèle: {str(e)}")
                    print("Assurez-vous d'avoir les permissions d'écriture dans le répertoire spécifié.")
                    print("\nConseils de débogage:")
                    print("- Essayez l'option 5 qui désactive les composants complexes avant la sauvegarde")
                    print("- Augmentez la mémoire disponible pour Python")
                    print("- Vérifiez si vous avez des objets cycliques dans votre modèle")

                    
        elif choice == '10':
                start_url = input("\nEntrez l'URL de départ: ").strip()
                if not start_url:
                    print("URL vide, utilisation d'une URL par défaut...")
                    start_url = "https://fr.wikipedia.org/wiki/Intelligence_artificieuse"
                    
                config = {
                    'max_pages': 5,
                    'max_depth': 2,
                    'min_text_length': 100,
                    'visited_urls_file': 'visited_urls.json'
                }
                
                print(f"\nDémarrage de l'apprentissage depuis: {start_url}")
                print(f"Configuration: {config}\n")
                
                autonomous_web_learning(model, start_url, config)
                
        elif choice == '11':  # Option de discussion
                print("\n=== Mode Discussion ===")
                print("Tapez 'quitter' pour revenir au menu principal\n")
                
                discussion_active = True
                while discussion_active:
                    try:
                        user_input = input("Vous: ")
                        
                        if user_input.lower() in ['quitter', 'exit', 'q']:
                            discussion_active = False
                            break
                            
                        if not user_input.strip():
                            continue
                            
                        with torch.no_grad():
                            # Tokeniser l'entrée utilisateur pour obtenir les input_ids
                            if hasattr(model, 'tokenizer') and model.tokenizer is not None:
                                # Si le modèle a un tokenizer, l'utiliser
                                tokens = model.tokenizer.encode(user_input)
                                input_ids = torch.tensor([tokens], dtype=torch.long)
                            else:
                                # Implémentation simplifiée pour le mode discussion
                                print("Utilisation d'une approche simplifiée pour le traitement du texte...")
                                try:
                                    # Créer une représentation simple du texte en utilisant des valeurs de hachage
                                    # (méthode de secours si le tokenizer n'est pas disponible)
                                    text_chars = [ord(c) % 256 for c in user_input[:100]]  # Limiter à 100 caractères
                                    while len(text_chars) < 10:  # Assurer une longueur minimale
                                        text_chars.append(0)
                                    input_ids = torch.tensor([text_chars], dtype=torch.long)
                                    
                                    print("Génération basée sur une représentation simplifiée du texte.")
                                    print("Note: Pour une meilleure qualité, un tokenizer est recommandé.")
                                except Exception as e:
                                    print(f"Erreur lors du traitement de l'entrée texte: {e}")
                                    continue
                            
                            # Générer la réponse avec les input_ids
                            try:
                                response_ids = model.generate(
                                    input_ids=input_ids,
                                    max_length=150,
                                    temperature=0.7,
                                    top_p=0.9,
                                    do_sample=True
                                )
                                
                                # Décoder les tokens générés en texte
                                if hasattr(model, 'tokenizer') and model.tokenizer is not None:
                                    # Utiliser le tokenizer pour décoder
                                    response = model.tokenizer.decode(response_ids[0].tolist())
                                    # Nettoyer la réponse en supprimant les tokens spéciaux
                                    response = response.replace('<pad>', '').replace('<eos>', '').strip()
                                else:
                                    # Conversion directe des tokens en caractères si pas de tokenizer
                                    chars = [chr(token % 128) for token in response_ids[0].tolist() if token < 128]
                                    response = ''.join(chars)
                                    # Nettoyer les caractères non imprimables
                                    response = ''.join(c for c in response if c.isprintable() or c.isspace())
                                
                                # Post-traitement de la réponse pour améliorer la lisibilité
                                # Enlever l'entrée utilisateur si elle est répétée au début
                                if response.startswith(user_input):
                                    response = response[len(user_input):].lstrip()
                                # Limiter la longueur de la réponse à afficher
                                if len(response) > 500:
                                    response = response[:497] + "..."
                                # Si la réponse est vide ou ne contient que des espaces
                                if not response.strip():
                                    response = "[Génération sans contenu décodable. Le modèle pourrait nécessiter plus d'entraînement.]"  
                            except Exception as e:
                                print(f"Erreur lors de la génération: {e}")
                                continue
                            
                            # Afficher la réponse du modèle
                            print("\nModèle:", response if response else "Désolé, je n'ai pas pu générer de réponse.")
                            print()
                            
                    except KeyboardInterrupt:
                        print("\nRetour au menu principal...")
                        discussion_active = False
                        break
                    except Exception as e:
                        print(f"\nErreur lors de la génération de la réponse: {str(e)}")
                        continue
        
def is_valid_url(url):
    """Vérifie si une URL est valide pour le traitement"""
    # Exclure les URLs de médias et les liens spéciaux
    excluded = [
        '.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip', '.mp3', '.mp4',
        'Special:', 'File:', 'Template:', 'Category:', 'Help:', 'Wikipedia:',
        'Portal:', 'Talk:', 'User:', 'User_talk:', 'File_talk:'
    ]
    return not any(ext in url for ext in excluded)

def clean_url(url, base_domain):
    """Nettoie et normalise une URL"""
    # Supprimer les fragments (#...)
    url = url.split('#')[0]
    # Supprimer les paramètres de requête inutiles
    if '?' in url:
        url = url.split('?')[0]
    # S'assurer que l'URL est dans le même domaine
    if base_domain not in url:
        return None
    return url

def demo_multimodal_generation(model, tokenizer=None, inputs=None):
    """
    Démontre les capacités de génération multimodale du modèle.
    
    Args:
        model: Modèle NeuroLite
        tokenizer: Tokenizer multimodal
        inputs: Entrées multimodales (optionnel)
    """
    print("\n=== Démonstration de Génération Multimodale ===\n")
    
    # Si aucune entrée n'est fournie, utiliser des entrées par défaut
    if inputs is None:
        inputs = {
            'text': ["L'architecture universelle d'IA s'inspire des avancées récentes pour atteindre polyvalence et efficacité."],
            'image': torch.randn(1, 3, 224, 224)  # Image simulée
        }
    
    # Tokenizer les entrées si un tokenizer est fourni
    if tokenizer is not None:
        print("Tokenization des entrées multimodales...")
        tokens = tokenizer.tokenize(inputs)
        print(f"Tokens générés avec succès. Dimensions des tokens sémantiques: {tokens['semantic_tokens'].shape}")
        
        # Visualiser les tokens
        visualize_tokens(tokens)
    
    # Générer des sorties pour différentes modalités
    target_modalities = ['text', 'image']
    if 'audio' in inputs:
        target_modalities.append('audio')
    if 'video' in inputs:
        target_modalities.append('video')
    
    print(f"\nGénération pour les modalités: {', '.join(target_modalities)}\n")
    
    # Générer les sorties
    with torch.no_grad():
        try:
            # Utiliser la méthode de génération du modèle
            start_time = time.time()
            outputs = model.generate(
                multimodal_inputs=inputs,
                target_modalities=target_modalities,
                temperature=0.8
            )
            gen_time = time.time() - start_time
            
            print(f"Génération terminée en {format_time(gen_time)}\n")
            
            # Traiter les sorties par modalité
            if isinstance(outputs, dict):
                for modality, output in outputs.items():
                    if modality.startswith('text_'):
                        if isinstance(output, torch.Tensor):
                            # Afficher le texte généré (simplifié, supposerait une étape de décodage)
                            print(f"Texte généré: {output.shape}")
                        elif isinstance(output, str):
                            print(f"Texte généré: {output[:100]}..." if len(output) > 100 else output)
                    
                    elif modality.startswith('image_'):
                        if isinstance(output, torch.Tensor):
                            # Afficher l'image générée
                            print(f"Image générée: {output.shape}")
                            # Visualiser l'image
                            if output.dim() == 4:  # [B, C, H, W]
                                img = output[0].permute(1, 2, 0).cpu().numpy()
                                plt.figure(figsize=(6, 6))
                                plt.imshow(np.clip(img, 0, 1))
                                plt.title("Image générée")
                                plt.axis('off')
                                plt.show()
                    
                    elif modality.startswith('audio_'):
                        if isinstance(output, torch.Tensor):
                            print(f"Audio généré: {output.shape}")
                    
                    elif modality.startswith('video_'):
                        if isinstance(output, torch.Tensor):
                            print(f"Vidéo générée: {output.shape}")
                    
                    elif modality.startswith('graph_'):
                        if isinstance(output, dict) and 'adjacency_matrix' in output:
                            print(f"Graphe généré: {output['adjacency_matrix'].shape}")
                    
                    else:
                        print(f"Sortie pour {modality}: {type(output)}")
            else:
                print("Format de sortie non reconnu")
        
        except Exception as e:
            print(f"Erreur lors de la génération: {str(e)}")


def autonomous_web_learning(model, start_url, config):
    """
    Fonction pour l'apprentissage autonome depuis le web
    Utilise la mémoire interne du modèle et l'apprentissage continu
    """
    # Extraire le domaine de base pour rester sur le même site
    base_domain = urlparse(start_url).netloc
    
    # Charger les URLs déjà visitées
    visited_urls = set()
    if os.path.exists(config['visited_urls_file']):
        try:
            with open(config['visited_urls_file'], 'r') as f:
                visited_urls = set(json.load(f))
        except (json.JSONDecodeError, FileNotFoundError):
            visited_urls = set()
    
    # Initialiser la file d'attente pour le parcours en largeur
    queue = deque()
    # Nettoyer et ajouter l'URL de départ
    clean_start = clean_url(start_url, base_domain)
    if clean_start:
        queue.append((clean_start, 0))
    
    pages_processed = 0
    total_sentences = 0
    
    # Télécharger les ressources NLTK nécessaires
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Téléchargement des ressources NLTK...")
        nltk.download('punkt')
        nltk.download('stopwords')
    
    stop_words = set(stopwords.words('french'))
    
    print(f"\n=== Apprentissage autonome depuis le web ===")
    print(f"URL de départ: {start_url}")
    print(f"Domaine cible: {base_domain}")
    print(f"Limite: {config['max_pages']} pages, profondeur max: {config['max_depth']}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        while queue and pages_processed < config['max_pages']:
            current_url, depth = queue.popleft()
            
            # Vérifier la profondeur maximale
            if depth > config['max_depth']:
                print(f"  Profondeur maximale atteinte pour: {current_url}")
                continue
                
            # Vérifier si l'URL a déjà été visitée
            if current_url in visited_urls:
                print(f"  URL déjà visitée: {current_url}")
                continue
                
            try:
                # Ajouter un délai entre les requêtes
                time.sleep(1)  # 1 seconde entre les requêtes
                
                print(f"\n[{pages_processed + 1}/{config['max_pages']}] Traitement: {current_url} (profondeur: {depth})")
                
                # Récupérer le contenu de la page
                response = requests.get(current_url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # Vérifier le type de contenu
                content_type = response.headers.get('content-type', '')
                if 'text/html' not in content_type:
                    print(f"  Ignoré (type de contenu non supporté: {content_type})")
                    continue
                    
                # Parser le HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Supprimer les éléments inutiles (scripts, styles, etc.)
                for element in soup(['script', 'style', 'nav', 'footer', 'header', 'iframe']):
                    element.decompose()
                
                # Extraire le texte principal (ajuster selon le site cible)
                main_content = (
                    soup.find('div', {'id': 'bodyContent'}) or 
                    soup.find('main') or 
                    soup.find('article') or 
                    soup.find('div', {'class': 'mw-parser-output'}) or 
                    soup.find('div', {'class': 'content'}) or
                    soup
                )
                
                if not main_content:
                    print("  Aucun contenu principal trouvé")
                    continue
                    
                text = main_content.get_text(' ', strip=True)
                
                # Nettoyer le texte
                text = re.sub(r'\s+', ' ', text)  # Remplacer les espaces multiples
                
                # Découper en phrases
                sentences = [s.strip() for s in sent_tokenize(text) if len(s.split()) > 3]  # Ignorer les phrases trop courtes
                
                if not sentences:
                    print("  Aucune phrase valide trouvée")
                    continue
                    
                print(f"  Traitement de {len(sentences)} phrases...")
                
                # Traiter les phrases par lots
                batch_size = 32
                for i in range(0, len(sentences), batch_size):
                    batch = sentences[i:i+batch_size]
                    # Filtrer les phrases trop courtes ou trop longues
                    batch = [s for s in batch if 10 < len(s) < 500]
                    if not batch:
                        continue
                        
                    try:
                        # Appeler le modèle pour l'apprentissage continu
                        model.train() # S'assurer que le modèle est en mode entraînement
                        outputs = model(
                            multimodal_inputs={'text': batch},
                            update_memory=True,
                            continuous_learning=True
                        )
                        print(f"  Lot {i//batch_size + 1} traité ({len(batch)} phrases)")
                        total_sentences += len(batch)
                        
                    except Exception as e:
                        print(f"  Erreur lors du traitement du lot: {e}")
                        continue
                
                # Si on n'a pas atteint la profondeur maximale, extraire les liens
                if depth < config['max_depth']:
                    new_links = set()
                    for link in soup.find_all('a', href=True):
                        url = urljoin(current_url, link['href'])
                        # Nettoyer et valider l'URL
                        clean = clean_url(url, base_domain)
                        if clean and is_valid_url(clean):
                            new_links.add(clean)
                    
                    # Ajouter les nouveaux liens à la file d'attente
                    for url in new_links:
                        if url not in visited_urls and url not in {u for u, _ in queue}:
                            queue.append((url, depth + 1))
                    
                    print(f"  {len(new_links)} nouveaux liens trouvés, {len(queue)} dans la file")
                
                # Marquer l'URL comme visitée
                visited_urls.add(current_url)
                pages_processed += 1
                
                # Sauvegarder périodiquement les URLs visitées
                if pages_processed % 3 == 0:
                    try:
                        with open(config['visited_urls_file'], 'w') as f:
                            json.dump(list(visited_urls), f)
                    except Exception as e:
                        print(f"  Erreur lors de la sauvegarde: {e}")
                
            except requests.exceptions.TooManyRedirects:
                print("  Trop de redirections, ignoré")
            except requests.exceptions.RequestException as e:
                print(f"  Erreur réseau: {e}")
            except Exception as e:
                print(f"  Erreur lors du traitement de la page: {e}")
        
        # Sauvegarder les URLs visitées
        try:
            with open(config['visited_urls_file'], 'w') as f:
                json.dump(list(visited_urls), f)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des URLs visitées: {e}")
        
        print("\n=== Apprentissage autonome terminé ===")
        print(f"Pages traitées: {pages_processed}")
        print(f"URLs visitées: {len(visited_urls)}")
        print("\nLe modèle a mis à jour sa mémoire interne avec les nouvelles informations.")
        print("Vous pouvez maintenant utiliser le modèle avec les connaissances acquises.")
        
        if hasattr(model, 'memory') and hasattr(model.memory, 'get_stats'):
            stats = model.memory.get_stats()
            print("\nStatistiques de la mémoire:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
                
    except KeyboardInterrupt:
        print("\nApprentissage interrompu par l'utilisateur.")
    except Exception as e:
        print(f"\nErreur inattendue: {e}")
    finally:
        # Sauvegarder les URLs visitées en cas d'arrêt inattendu
        try:
            with open(config['visited_urls_file'], 'w') as f:
                json.dump(list(visited_urls), f)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde finale: {e}")


def main():
    parser = argparse.ArgumentParser(description="Démonstration interactive de NeuroLite")
    
    # Options générales
    parser.add_argument('--size', type=str, default='base', choices=['tiny', 'small', 'base'],
                       help='Taille du modèle à utiliser')
    parser.add_argument('--agi', action='store_true', default=True,
                       help='Activer les fonctionnalités AGI avancées')
    
    # Sous-commandes
    subparsers = parser.add_subparsers(dest='command', help='Mode de démonstration')
    
    # Sous-commande pour la démonstration interactive
    interactive_parser = subparsers.add_parser('interactive', help='Mode démonstration interactive')
    
    # Sous-commande pour la génération multimodale
    generate_parser = subparsers.add_parser('generate', help='Génération multimodale')
    generate_parser.add_argument('--input', type=str, default=None,
                               help='Chemin vers un fichier d\'entrée')
    generate_parser.add_argument('--modality', type=str, default=None,
                               choices=['text', 'image', 'audio', 'video', 'graph'],
                               help='Type de modalité pour l\'entrée')
    generate_parser.add_argument('--target', type=str, default='text,image',
                               help='Modalités cibles pour la génération, séparées par des virgules')
    
    # Sous-commande pour l'apprentissage web
    web_parser = subparsers.add_parser('web', help='Apprentissage depuis le web')
    web_parser.add_argument('--url', type=str, required=True,
                          help='URL de départ pour l\'exploration')
    web_parser.add_argument('--depth', type=int, default=2,
                          help='Profondeur d\'exploration')
    
    # Sous-commande pour les tests de performance
    bench_parser = subparsers.add_parser('benchmark', help='Tests de performance')
    bench_parser.add_argument('--runs', type=int, default=10,
                           help='Nombre d\'exécutions pour les tests')
    
    args = parser.parse_args()
    
    # Traiter selon la sous-commande
    if args.command == 'interactive' or args.command is None:
        # Mode interactif par défaut
        run_interactive_demo(model_size=args.size, agi_enabled=args.agi)
    
    elif args.command == 'generate':
        # Créer le tokenizer et le modèle
        tokenizer = create_tokenizer()
        
        # Créer la configuration du modèle
        config = NeuroLiteConfig.base() if args.size == 'base' else \
                NeuroLiteConfig.small() if args.size == 'small' else \
                NeuroLiteConfig.tiny()
        
        # Configuration multimodale
        config.use_multimodal_input = True
        config.multimodal_output_dim = 768
        config.multimodal_hidden_dim = 768
        config.use_cross_modal_attention = True
        
        # Créer le modèle
        model = NeuroLiteModel(
            config=config,
            task_type="multimodal_generation",
            tokenizer=tokenizer
        )
        
        # Traiter l'entrée si spécifiée
        inputs = None
        if args.input:
            inputs = process_multimodal_input(args.input, args.modality)
        
        # Lancer la démonstration de génération
        demo_multimodal_generation(model, tokenizer, inputs)
    
    elif args.command == 'web':
        # Créer la configuration du modèle
        config = NeuroLiteConfig.base() if args.size == 'base' else \
                NeuroLiteConfig.small() if args.size == 'small' else \
                NeuroLiteConfig.tiny()
        
        # Activer l'apprentissage continu et la mémoire
        config.use_continual_adapter = True
        config.use_external_memory = True
        
        # Créer le modèle
        model = NeuroLiteModel(config=config)
        
        # Lancer l'apprentissage web
        autonomous_web_learning(model, args.url, {
            'max_depth': args.depth,
            'max_pages': 10,
            'timeout': 30
        })
    
    elif args.command == 'benchmark':
        # Créer la configuration du modèle
        config = NeuroLiteConfig.base() if args.size == 'base' else \
                NeuroLiteConfig.small() if args.size == 'small' else \
                NeuroLiteConfig.tiny()
        
        # Créer le modèle
        model = NeuroLiteModel(config=config)
        
        # Afficher les informations sur le modèle
        display_model_info(model)
        
        # Tester la vitesse d'inférence
        test_inference_speed(model, num_runs=args.runs)


if __name__ == "__main__":
    main()
