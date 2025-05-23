"""
NeuroLite - Démonstration Interactive

Cette application montre les capacités de l'architecture NeuroLite,
une alternative légère aux Transformers pour les appareils contraints.
"""

import torch
import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from neurolite import NeuroLiteModel, NeuroLiteConfig, NeuroLiteForClassification
from neurolite import MultimodalProjection, HierarchicalMemory, NeurosymbolicReasoner, ContinualAdapter


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
    
    # Créer des données simulées
    if hasattr(model, 'input_projection') and hasattr(model.input_projection, '_compute_minhash_bloom'):
        # Projection pour texte brut
        inputs = ["Ceci est un exemple de texte pour tester la vitesse d'inférence de NeuroLite."]
        inference_func = lambda: model(input_texts=inputs)
    else:
        # Projection tokenisée 
        seq_len = 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
        inference_func = lambda: model(input_ids=input_ids)
    
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
    
    # Activer tous les modules standards
    config.use_external_memory = True
    config.use_dynamic_routing = True
    config.use_symbolic_module = True
    
    # Activer les modules AGI avancés
    config.use_hierarchical_memory = True
    config.short_term_memory_size = 64
    config.long_term_memory_size = 128
    config.persistent_memory_size = 256
    
    # Mécanismes de raisonnement avancé
    config.use_advanced_reasoning = True
    config.symbolic_dim = 64
    config.num_inference_steps = 3
    
    # Planification structurée
    config.use_planning_module = True
    config.plan_dim = 32
    
    # Apprentissage continu
    config.use_continual_learning = True
    
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
        print("1. Traiter un nouveau texte")
        print("2. Comparer deux textes (similarité)")
        print("3. Démonstration de la mémoire contextuelle")
        print("4. Démonstration du raisonnement symbolique")
        print("5. Démonstration de la planification structurée")  
        print("6. Démonstration de l'apprentissage continu")
        print("7. Afficher les informations du modèle")
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
                outputs = model(input_texts=[text])
                
            process_time = time.time() - start_time
            
            # Ajouter au contexte de session
            session_context.append(text)
            if len(session_context) > 5:
                session_context.pop(0)
            
            # Obtenir l'embedding moyen
            embedding = torch.mean(outputs, dim=1).squeeze().numpy()
            
            print(f"Temps de traitement: {format_time(process_time)}")
            print(f"Dimension de sortie: {outputs.shape}")
            print(f"Norme de l'embedding: {np.linalg.norm(embedding):.4f}")
            
            # Afficher les premiers éléments de l'embedding
            print("\nAperçu de l'embedding:")
            preview = embedding[:5]
            print(", ".join([f"{x:.4f}" for x in preview]))
            
        elif choice == '2':
            print("\nEntrez deux textes à comparer:")
            text1 = input("Texte 1: ")
            text2 = input("Texte 2: ")
            
            if not text1 or not text2:
                continue
                
            print("Calcul de la similarité...")
            
            with torch.no_grad():
                output1 = model(input_texts=[text1])
                output2 = model(input_texts=[text2])
                
                # Calculer les embeddings moyens
                emb1 = torch.mean(output1, dim=1).squeeze()
                emb2 = torch.mean(output2, dim=1).squeeze()
                
                # Normaliser
                emb1 = emb1 / torch.norm(emb1)
                emb2 = emb2 / torch.norm(emb2)
                
                # Calculer la similarité cosinus
                similarity = torch.dot(emb1, emb2).item()
            
            print(f"Similarité cosinus: {similarity:.4f}")
            
            # Interprétation qualitative
            if similarity > 0.9:
                print("Interprétation: Textes très similaires")
            elif similarity > 0.7:
                print("Interprétation: Textes assez similaires")
            elif similarity > 0.5:
                print("Interprétation: Textes modérément similaires")
            elif similarity > 0.3:
                print("Interprétation: Textes peu similaires")
            else:
                print("Interprétation: Textes très différents")
                
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
                no_context_output = model(input_texts=[query], update_memory=False)
                no_context_emb = torch.mean(no_context_output, dim=1).squeeze()
            
            # 2. Avec contexte (nourrir la mémoire avec le contexte)
            if hasattr(model.memory, 'initialized'):
                model.memory.initialized = False
                
            with torch.no_grad():
                # Traiter le contexte
                for ctx in session_context:
                    _ = model(input_texts=[ctx], update_memory=True)
                    
                # Traiter la requête
                context_output = model(input_texts=[query], update_memory=False)
                context_emb = torch.mean(context_output, dim=1).squeeze()
            
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
                    input_texts=[premise],
                    external_facts=facts,
                    return_symbolic=True
                )
                
                # Traiter l'hypothèse avec retour des informations symboliques
                hypothesis_result = model(
                    input_texts=[hypothesis],
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
            # Afficher les informations sur le modèle
            display_model_info(model)
            
            # Tester à nouveau la vitesse d'inférence
            avg_time = test_inference_speed(model)
            
        else:
            print("Option non reconnue.")
    
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
