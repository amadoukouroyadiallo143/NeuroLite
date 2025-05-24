"""
NeuroLite - Démonstration Interactive

Cette application montre les capacités de l'architecture NeuroLite,
une architecture universelle et légère conçue pour des applications
multimodales et l'intelligence artificielle générale.
"""

import torch
import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from neurolite import (
    NeuroLiteModel, 
    NeuroLiteConfig,
    MultimodalProjection,
    CrossModalAttention,
    HierarchicalMemory,
    VectorMemoryStore,
    NeurosymbolicReasoner,
    StructuredPlanner,
    ContinualAdapter,
    ReplayBuffer,
    ProgressiveCompressor,
    BayesianBeliefNetwork
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


def run_interactive_demo(model_size='tiny'):
    """Lance une démonstration interactive"""
    print(f"Chargement du modèle NeuroLite ({model_size})...")
    
    # Créer la configuration
    if model_size == 'tiny':
        config = NeuroLiteConfig.tiny()
    elif model_size == 'small':
        config = NeuroLiteConfig.small()
    else:
        config = NeuroLiteConfig.base()
        
    # Activer l'apprentissage continu
    config.use_continual_adapter = True
    config.continual_adapter_buffer_size = 1000
    
    # Configuration multimodale
    config.use_multimodal_input = True
    config.multimodal_output_dim = 512
    config.multimodal_image_patch_size = 16
    config.multimodal_video_num_sampled_frames = 8
    
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
        print("\nOptions:")
        print("1. Traiter un texte")
        print("2. Analyser une image")
        print("3. Traiter une séquence vidéo")
        print("4. Fusion multimodale (texte + image)")
        print("5. Démonstration de la mémoire hiérarchique")
        print("6. Raisonnement symbolique avancé")
        print("7. Planification et réseau bayésien")
        print("8. Apprentissage continu et compression")
        print("9. Afficher les informations du modèle")
        print("0. Quitter")
        
        choice = input("\nVotre choix: ")
        
        if choice == '0' or choice.lower() in ['exit', 'quit']:
            break
            
        elif choice == '1':
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
            
        elif choice == '2':
            print("\nAnalyse d'image...")
            image_path = input("Entrez le chemin de l'image: ")
            if not os.path.exists(image_path):
                print("Fichier non trouvé!")
                continue
                
            # Charger et prétraiter l'image
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224, 224))
            image_tensor = torch.FloatTensor(np.array(image)).permute(2, 0, 1).unsqueeze(0)
            
            print("Analyse en cours...")
            with torch.no_grad():
                outputs = model(multimodal_inputs={'image': image_tensor})
                
            # Afficher les résultats
            embedding = outputs['hidden_states'].squeeze().numpy()
            print(f"\nDimension de l'embedding: {embedding.shape}")
            print(f"Norme de l'embedding: {np.linalg.norm(embedding):.4f}")
            
            # Afficher les premiers éléments de l'embedding
            print("\nAperçu de l'embedding:")
            preview = embedding[:5]
            print(", ".join([f"{x:.4f}" for x in preview]))
            
        elif choice == '3':
            print("\nTraitement vidéo...")
            print("Simulation d'une séquence de 8 frames...")
            
            # Simuler une séquence vidéo
            video_tensor = torch.randn(1, 8, 3, 224, 224)
            
            with torch.no_grad():
                outputs = model(multimodal_inputs={'video': video_tensor})
                
            # Afficher les résultats
            video_emb = outputs['hidden_states'].squeeze().numpy()
            print(f"\nDimension de l'embedding vidéo: {video_emb.shape}")
            
        elif choice == '4':
            print("\nFusion multimodale...")
            text = input("Entrez une description: ")
            image_path = input("Entrez le chemin de l'image: ")
            
            if not os.path.exists(image_path):
                print("Fichier image non trouvé!")
                continue
                
            # Préparer les entrées
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224, 224))
            image_tensor = torch.FloatTensor(np.array(image)).permute(2, 0, 1).unsqueeze(0)
            
            print("Analyse multimodale en cours...")
            with torch.no_grad():
                outputs = model(
                    multimodal_inputs={
                        'text': [text],
                        'image': image_tensor
                    }
                )
                
            # Afficher les résultats
            # Les embeddings sont dans hidden_states
            embeddings = outputs['hidden_states'].squeeze().numpy()
            text_emb = embeddings[:128]  # Premiers 128 pour le texte
            image_emb = embeddings[128:256]  # 128 suivants pour l'image
            fused_emb = embeddings[256:]  # Le reste pour la fusion
            
            print("\nRésultats de la fusion:")
            print(f"Dimension texte: {text_emb.shape}")
            print(f"Dimension image: {image_emb.shape}")
            print(f"Dimension fusion: {fused_emb.shape}")
            
            # Calculer les similarités
            text_image_sim = torch.nn.functional.cosine_similarity(
                torch.from_numpy(text_emb).unsqueeze(0),
                torch.from_numpy(image_emb).unsqueeze(0)
            ).item()
            
            print(f"\nSimilarité texte-image: {text_image_sim:.4f}")
            
        elif choice == '5':
            print("\nDémonstration de la mémoire hiérarchique...")
            
            # Obtenir les statistiques de mémoire
            print("\nÉtat de la mémoire:")
            
            # Court terme
            short_term_size = model.memory.short_term_memory.memory_keys.size(1)
            short_term_active = (model.memory.short_term_memory.memory_age > 0).sum().item()
            print(f"Mémoire court terme: {short_term_active}/{short_term_size} éléments actifs")
            
            # Long terme
            long_term_size = model.memory.long_term_memory.memory_keys.size(1)
            long_term_active = (model.memory.long_term_memory.memory_age > 0).sum().item()
            print(f"Mémoire long terme: {long_term_active}/{long_term_size} éléments actifs")
            
            # Persistante
            persistent_size = model.memory.persistent_memory.memory_keys.size(1)
            persistent_active = int(model.memory.persistent_memory.memory_usage.sum().item())
            print(f"Mémoire persistante: {persistent_active}/{persistent_size} éléments actifs")
            
            # Taux d'utilisation global
            total_active = short_term_active + long_term_active + persistent_active
            total_size = short_term_size + long_term_size + persistent_size
            utilization = total_active / total_size if total_size > 0 else 0
            print(f"Taux d'utilisation global: {utilization:.2%}")
            
            # Démonstration de la recherche
            query_text = input("\nEntrez une requête: ")
            if query_text:
                print("Génération de l'embedding pour la requête...")
                with torch.no_grad():
                    # Utiliser le modèle principal pour obtenir l'embedding de la requête
                    outputs_dict = model(multimodal_inputs={'text': [query_text]})
                    query_embedding_sequence = outputs_dict['hidden_states'] # Attendue: [batch_size, seq_len, hidden_size]
                    
                    # Convertir la séquence d'embedding en un seul vecteur pour la recherche
                    # (ex: moyenne des embeddings de tokens, ou embedding du token [CLS])
                    # Pour l'instant, utilisons la moyenne sur la dimension de séquence si elle existe
                    if query_embedding_sequence.dim() == 3 and query_embedding_sequence.size(1) > 1:
                        query_embedding = query_embedding_sequence.mean(dim=1) # [batch_size, hidden_size]
                    elif query_embedding_sequence.dim() == 3 and query_embedding_sequence.size(1) == 1:
                        query_embedding = query_embedding_sequence.squeeze(1) # [batch_size, hidden_size]
                    elif query_embedding_sequence.dim() == 2:
                        query_embedding = query_embedding_sequence # Déjà [batch_size, hidden_size] ou [hidden_size]
                    else:
                        print("Format d'embedding de requête inattendu.")
                        continue # Passer à l'itération suivante de la boucle principale
                    
                    # S'assurer que query_embedding est sur le bon device
                    model_device = next(model.parameters()).device
                    query_embedding = query_embedding.to(model_device)

                print("Recherche dans tous les niveaux...")
                with torch.no_grad():
                    results = model.memory.search(
                        query_embedding=query_embedding, # Passer l'embedding
                        k=3  # Top-3 résultats
                    )
                    
                print("\nRésultats par niveau:")
                for level, matches in results.items():
                    print(f"\n{level}:")
                    for score, item in matches:
                        print(f"[{score:.4f}] {item}")
                        
        elif choice == '6':
            print("\nRaisonnement symbolique avancé...")
            
            # Démonstration du raisonnement
            context = input("Entrez un contexte: ")
            query = input("Entrez une question: ")
            
            if context and query:
                print("\nAnalyse en cours...")
                with torch.no_grad():
                    # Combiner le contexte et la question en une seule entrée pour le modèle principal
                    combined_text = context + f" [SEP] {query}" # Utiliser un séparateur clair
                    
                    # 1. Obtenir les états cachés du modèle principal pour le texte combiné
                    main_model_outputs = model(multimodal_inputs={'text': [combined_text]})
                    input_hidden_states_for_reasoner = main_model_outputs['hidden_states'] # [batch_size, seq_len, hidden_size]
                    
                    # S'assurer que les états cachés sont sur le bon device
                    model_device = next(model.parameters()).device
                    input_hidden_states_for_reasoner = input_hidden_states_for_reasoner.to(model_device)

                    # 2. Appeler le module de raisonnement symbolique (qui appelle son propre forward)
                    # Le NeurosymbolicReasoner.forward prend `hidden_states` et `external_facts` (optionnel)
                    # et `return_symbolic` (booléen)
                    reasoning_output, symbolic_details = model.symbolic(
                        hidden_states=input_hidden_states_for_reasoner,
                        return_symbolic=True 
                    )
                
                print("\nRésultat du raisonnement (état caché final après raisonnement):")
                print(f"  Shape: {reasoning_output.shape}")
                if reasoning_output.numel() > 0:
                    print(f"  Aperçu: {reasoning_output.flatten()[:5].tolist()}...")

                # Afficher l'interprétation des résultats symboliques
                print("\n" + "="*50)
                print("INTERPRÉTATION DES RÉSULTATS SYMBOLIQUES")
                print("="*50)
                interpretation = interpret_symbolic_results(symbolic_details)
                print(interpretation)
                
                # Afficher les détails bruts (optionnel, pour le débogage)
                print("\n" + "-"*50)
                print("DÉTAILS TECHNIQUES (pour débogage)")
                print("-"*50)
                if symbolic_details:
                    for key, value in symbolic_details.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                            if value.numel() > 0 and value.numel() < 50:
                                print(f"     Aperçu: {value.flatten()[:10].tolist()}")
                            elif value.numel() > 0:
                                print(f"     Aperçu (premiers éléments): {value.flatten()[:5].tolist()}...")
                        else:
                            print(f"  {key}: {value}")
                else:
                    print("  Aucun détail symbolique retourné.")
                print("="*50 + "\n")
            else:
                print("Contexte et question non fournis.")
                
        elif choice == '7':
            print("\nPlanification et réseau bayésien...")
            
            # Définir un objectif
            goal = input("Définissez un objectif: ")
            if goal:
                print("\nGénération du plan...")
                with torch.no_grad():
                    plan = model.planner.generate_plan(goal=goal)
                    
                print("\nPlan structuré:")
                for i, step in enumerate(plan['steps'], 1):
                    print(f"\n{i}. {step['action']}")
                    print(f"   Probabilité: {step['probability']:.2f}")
                    if 'dependencies' in step:
                        print(f"   Dépendances: {', '.join(step['dependencies'])}")
                        
                # Mise à jour du réseau bayésien
                # Convertir l'objectif en format tensor
                goal_tensor = torch.zeros(model.bayesian_network.num_variables, device=next(model.parameters()).device)
                # Simple mapping d'objectif -> variable (à améliorer)
                goal_tensor[hash(goal) % model.bayesian_network.num_variables] = 1.0
                
                if hasattr(model.bayesian_network, '_infer_probabilities'):
                    model.bayesian_network._infer_probabilities(goal_tensor.unsqueeze(0))  # Ajout batch dimension
                else:
                    print("Avertissement: Méthode update non disponible")
                
                print("\nProbabilités mises à jour:")
                beliefs = model.bayesian_network.get_beliefs()
                for var, prob in beliefs.items():
                    print(f"{var}: {prob:.2f}")
                    
        elif choice == '8':
            print("\nApprentissage continu et compression...\n")
            
            if hasattr(model, 'continual_adapter') and model.continual_adapter is not None:
                print("Statistiques d'apprentissage:")
                print(f"Taux d'adaptation: {model.continual_adapter.adaptation_rate}")
                
                compress = input("Lancer une compression? (o/n) ").strip().lower() == 'o'
                if compress:
                    print("Compression en cours...")
                    if hasattr(model, 'progressive_compressor') and model.progressive_compressor is not None:
                        # Exemple avec des données réelles
                        sample_input = {
                            'text': ["Exemple de texte à compresser"],
                            'image': torch.randn(1, 3, 224, 224) if hasattr(model, 'input_projection') else None
                        }
                        with torch.no_grad():
                            outputs = model(multimodal_inputs=sample_input)
                            hidden_states = outputs['hidden_states']
                            compressed = model.progressive_compressor(hidden_states)
                            if isinstance(compressed, tuple):
                                compressed = compressed[0]  # Take first element if output is tuple
                            print(f"Taille avant: {hidden_states.shape}")
                            print(f"Taille après: {compressed.shape}")
                            print(f"Taux de compression: {model.progressive_compressor.compression_ratio:.1%}")
                    else:
                        print("Module de compression non configuré - vérifiez use_progressive_compression dans la config")
            else:
                print("Module d'apprentissage continu non configuré dans ce modèle.")
                
        elif choice == '9':
            display_model_info(model)
            test_inference_speed(model)
            
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
            
            # Calculer la différence normalisée
            no_context_norm = no_context_emb / torch.norm(no_context_emb)
            context_norm = context_emb / torch.norm(context_emb)
            similarity = torch.dot(no_context_norm, context_norm).item()
            
            # Plus la similarité est faible, plus l'impact du contexte est fort
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
    
    print("Merci d'avoir utilisé la démo NeuroLite!")


def main():
    parser = argparse.ArgumentParser(description="Démo de NeuroLite avec fonctionnalités AGI")
    parser.add_argument(
        "--size",
        choices=["tiny", "small", "base"],
        default="tiny",
        help="Taille du modèle à utiliser"
    )
    parser.add_argument(
        "--agi", 
        action="store_true",
        help="Activer toutes les fonctionnalités AGI avancées"
    )
    args = parser.parse_args()
    
    run_interactive_demo(args.size)


if __name__ == "__main__":
    main()
