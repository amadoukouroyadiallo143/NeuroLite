"""
Exemple démontrant l'utilisation de la mémoire hiérarchique et du routage dynamique de NeuroLite.
Ce script montre comment ces composants AGI permettent au modèle de:
1. Retenir de l'information contextuelle à plusieurs niveaux temporels (court, long terme et persistant)
2. Activer conditionnellement différents experts selon le type d'entrée
3. Démontrer le fonctionnement de la mémoire hiérarchique pour le traitement séquentiel
"""

import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from neurolite import NeuroLiteModel, NeuroLiteConfig, HierarchicalMemory


def analyze_routing_patterns(model, texts):
    """
    Analyse les patterns d'activation des experts dans le modèle
    pour différents types de textes.
    """
    model.eval()
    
    # Récupérer une référence aux couches de routage
    routing_layers = []
    for layer in model.layers:
        if hasattr(layer, 'router'):
            routing_layers.append(layer)
    
    if not routing_layers:
        print("Aucune couche de routage trouvée dans le modèle.")
        return
    
    # Collecter les activations des routeurs pour chaque texte
    activations = []
    
    with torch.no_grad():
        for text in texts:
            # Forward pass
            _ = model(input_texts=[text])
            
            # Pour la première couche de routage, extraire les poids d'activation
            layer = routing_layers[0]
            
            # Récupérer les scores de routage (logits avant softmax)
            # Pour simplifier, on utilise un hook temporaire pour capturer les scores
            router_scores = []
            
            def hook_fn(module, input, output):
                router_scores.append(output.detach().cpu())
            
            # Enregistrer le hook sur le routeur
            handle = layer.router.router.register_forward_hook(hook_fn)
            
            # Refaire le forward pass pour capturer les scores
            _ = model(input_texts=[text])
            
            # Retirer le hook
            handle.remove()
            
            # Appliquer softmax pour avoir les probabilités
            probs = nn.functional.softmax(router_scores[0][0, 0], dim=-1).numpy()
            activations.append(probs)
    
    return np.array(activations)


def test_hierarchical_memory(model, context_sequence, query_texts):
    """
    Teste la capacité de la mémoire hiérarchique à retenir et utiliser des
    informations à différentes échelles temporelles.
    
    Args:
        model: Modèle NeuroLite avec mémoire hiérarchique
        context_sequence: Liste ordonnée de textes formant une séquence narrative
        query_texts: Liste de requêtes liées à différentes parties du contexte
        
    Returns:
        Dictionary contenant les embeddings à différents niveaux de la mémoire
    """
    model.eval()
    
    print("\n=== Test de la mémoire hiérarchique ===\n")
    print("Cette démonstration montre comment la mémoire hiérarchique stocke l'information")
    print("dans les couches court, long terme et persistante en fonction de l'importance.\n")
    
    # Étape 1: Générer des embeddings pour les requêtes sans contexte mémoriel
    print("1. Génération des embeddings de référence (sans contexte)...")
    
    memory_states = {}
    memory_states["baseline"] = []
    
    with torch.no_grad():
        for query in query_texts:
            # Désactiver la mise à jour de la mémoire pour ce passage
            output = model(input_texts=[query], update_memory=False)
            
            # Prendre la moyenne sur la dimension de séquence pour simplifier
            embedding = torch.mean(output, dim=1).cpu().numpy()
            memory_states["baseline"].append(embedding[0])
    
    # Étape 2: Traiter la séquence de contexte en plusieurs phases pour démontrer
    # le transfert entre les différents niveaux de mémoire
    print("\n2. Alimentation de la mémoire hiérarchique (séquence narrative)...")
    
    # Diviser la séquence en 3 phases pour démontrer les différents niveaux de mémoire
    sequence_length = len(context_sequence)
    phase_size = sequence_length // 3
    
    # Phase 1: Remplir la mémoire à court terme
    print("\n   Phase 1: Remplissage de la mémoire à court terme...")
    with torch.no_grad():
        for i in range(phase_size):
            _ = model(input_texts=[context_sequence[i]], update_memory=True)
    
    # Capturer l'état après la phase 1
    memory_states["short_term"] = []
    with torch.no_grad():
        for query in query_texts:
            output = model(input_texts=[query], update_memory=False)
            embedding = torch.mean(output, dim=1).cpu().numpy()
            memory_states["short_term"].append(embedding[0])
    
    # Phase 2: Les données commencent à migrer vers la mémoire à long terme
    print("\n   Phase 2: Migration vers la mémoire à long terme...")
    with torch.no_grad():
        for i in range(phase_size, 2*phase_size):
            _ = model(input_texts=[context_sequence[i]], update_memory=True)
    
    # Capturer l'état après la phase 2
    memory_states["long_term"] = []
    with torch.no_grad():
        for query in query_texts:
            output = model(input_texts=[query], update_memory=False)
            embedding = torch.mean(output, dim=1).cpu().numpy()
            memory_states["long_term"].append(embedding[0])
    
    # Phase 3: Les données importantes migrent vers la mémoire persistante
    print("\n   Phase 3: Consolidation dans la mémoire persistante...")
    with torch.no_grad():
        for i in range(2*phase_size, sequence_length):
            _ = model(input_texts=[context_sequence[i]], update_memory=True)
    
    # Capturer l'état final
    memory_states["persistent"] = []
    with torch.no_grad():
        for query in query_texts:
            output = model(input_texts=[query], update_memory=False)
            embedding = torch.mean(output, dim=1).cpu().numpy()
            memory_states["persistent"].append(embedding[0])
    
    return memory_states


def test_memory_retention(model, context_texts, query_texts):
    """
    Teste la capacité de la mémoire externe à retenir et utiliser des
    informations contextuelles.
    
    Args:
        model: Modèle NeuroLite avec mémoire externe
        context_texts: Liste de textes servant de contexte à mémoriser
        query_texts: Liste de requêtes éventuellement liées au contexte
        
    Returns:
        Tuple contenant les embeddings sans contexte et avec contexte
    """
    model.eval()
    
    print("Test de rétention mémorielle...")
    
    # Étape 1: Générer des embeddings pour les requêtes sans contexte mémoriel
    # D'abord, réinitialiser la mémoire du modèle
    print("Génération des embeddings de référence (sans contexte)...")
    
    # Générer des embeddings sans contexte
    baseline_embeddings = []
    with torch.no_grad():
        for query in query_texts:
            # Désactiver la mise à jour de la mémoire pour ce passage
            output = model(input_texts=[query], update_memory=False)
            
            # Prendre la moyenne sur la dimension de séquence pour simplifier
            embedding = torch.mean(output, dim=1).cpu().numpy()
            baseline_embeddings.append(embedding[0])
    
    # Étape 2: Alimenter le modèle avec du contexte
    print("\nAlimentant la mémoire avec le contexte...")
    with torch.no_grad():
        for ctx_text in context_texts:
            # Mise à jour de la mémoire active
            _ = model(input_texts=[ctx_text], update_memory=True)
    
    # Étape 3: Générer à nouveau des embeddings pour les mêmes requêtes
    print("Génération des embeddings avec contexte mémoriel...")
    context_embeddings = []
    with torch.no_grad():
        for query in query_texts:
            # Cette fois-ci, la mémoire contient le contexte
            # mais on ne la met pas à jour avec les requêtes pour isoler l'effet
            output = model(input_texts=[query], update_memory=False)
            
            # Prendre la moyenne sur la dimension de séquence pour simplifier
            embedding = torch.mean(output, dim=1).cpu().numpy()
            context_embeddings.append(embedding[0])
    
    return np.array(baseline_embeddings), np.array(context_embeddings)


def plot_routing_heatmap(activations, text_labels, title):
    """Crée une heatmap des activations de routage"""
    plt.figure(figsize=(10, 6))
    plt.imshow(activations, cmap='viridis', aspect='auto')
    plt.colorbar(label='Probabilité d\'activation')
    plt.xlabel('Expert')
    plt.ylabel('Type de texte')
    plt.title(title)
    plt.yticks(np.arange(len(text_labels)), text_labels)
    plt.xticks(np.arange(activations.shape[1]), [f'Expert {i+1}' for i in range(activations.shape[1])])
    plt.tight_layout()
    plt.savefig('routing_heatmap.png')
    print("Heatmap de routage sauvegardée dans 'routing_heatmap.png'")


def plot_hierarchical_memory_effect(memory_states, query_labels):
    """
    Visualise l'effet de la mémoire hiérarchique sur les représentations.
    
    Args:
        memory_states: Dictionnaire contenant les embeddings pour chaque niveau de mémoire
        query_labels: Labels des requêtes pour la visualisation
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.manifold import TSNE
    
    # Créer une matrice de similarité entre les états de la mémoire hiérarchique
    memory_levels = ["baseline", "short_term", "long_term", "persistent"]
    memory_level_labels = ["Sans contexte", "Court terme", "Long terme", "Persistant"]
    
    # Couleurs pour chaque niveau de mémoire
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. Graphique de similarité entre les niveaux de mémoire
    print("\nCalcul des similarités entre niveaux de mémoire...")
    sim_matrix = np.zeros((len(query_labels), len(memory_levels), len(memory_levels)))
    
    for q_idx in range(len(query_labels)):
        for i, level1 in enumerate(memory_levels):
            for j, level2 in enumerate(memory_levels):
                sim = cosine_similarity(
                    [memory_states[level1][q_idx]], 
                    [memory_states[level2][q_idx]]
                )[0][0]
                sim_matrix[q_idx, i, j] = sim
    
    # Moyenne des similarités pour toutes les requêtes
    avg_sim_matrix = np.mean(sim_matrix, axis=0)
    
    # Créer une heatmap de la matrice de similarité moyenne
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_sim_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=memory_level_labels, yticklabels=memory_level_labels)
    plt.title("Similarité moyenne entre niveaux de mémoire")
    plt.tight_layout()
    plt.savefig("hierarchical_memory_similarity.png")
    print("Figure sauvegardée: 'hierarchical_memory_similarity.png'")
    
    # 2. Evolution des représentations à travers les niveaux de mémoire (t-SNE)
    print("\nGénération de la visualisation t-SNE des niveaux de mémoire...")
    
    # Combiner tous les embeddings pour la projection t-SNE
    all_embeddings = []
    embedding_labels = []
    level_indices = []
    query_indices = []
    
    for level_idx, level in enumerate(memory_levels):
        for query_idx in range(len(query_labels)):
            all_embeddings.append(memory_states[level][query_idx])
            embedding_labels.append(f"{level} - {query_labels[query_idx]}")
            level_indices.append(level_idx)
            query_indices.append(query_idx)
    
    # Convertir en array numpy
    all_embeddings = np.array(all_embeddings)
    
    # Appliquer t-SNE pour la réduction dimensionnelle
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Créer la visualisation
    plt.figure(figsize=(12, 10))
    
    # Tracer chaque niveau de mémoire avec une couleur différente
    for level_idx, level in enumerate(memory_levels):
        mask = np.array(level_indices) == level_idx
        plt.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1],
            color=colors[level_idx],
            label=memory_level_labels[level_idx],
            marker="o",
            s=100,
            alpha=0.7
        )
    
    # Tracer des lignes qui connectent la même requête à travers les niveaux de mémoire
    for q_idx in range(len(query_labels)):
        # Obtenir les indices de cette requête dans chaque niveau
        query_points = []
        for level_idx, _ in enumerate(memory_levels):
            indices = np.where((np.array(query_indices) == q_idx) & 
                             (np.array(level_indices) == level_idx))[0]
            if len(indices) > 0:
                query_points.append(indices[0])
        
        # Connecter les points avec des lignes
        for i in range(len(query_points) - 1):
            plt.plot(
                [embeddings_2d[query_points[i], 0], embeddings_2d[query_points[i+1], 0]],
                [embeddings_2d[query_points[i], 1], embeddings_2d[query_points[i+1], 1]],
                'k-', alpha=0.3
            )
        
        # Ajouter un label pour la dernière position (persistante)
        if len(query_points) > 0:
            plt.text(
                embeddings_2d[query_points[-1], 0] + 0.1,
                embeddings_2d[query_points[-1], 1] + 0.1,
                query_labels[q_idx],
                fontsize=9
            )
    
    plt.title("Evolution des représentations à travers les niveaux de mémoire")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hierarchical_memory_evolution.png")
    print("Figure sauvegardée: 'hierarchical_memory_evolution.png'")


def plot_memory_effect(baseline_embeddings, context_embeddings, query_labels):
    """
    Visualise l'effet de la mémoire sur les représentations.
    
    Args:
        baseline_embeddings: Embeddings sans contexte mémoriel
        context_embeddings: Embeddings avec contexte mémoriel
        query_labels: Labels des requêtes pour la visualisation
    """
    # Comparer la similarité cosinus entre les versions avec/sans contexte
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Calculer la similarité entre les embeddings baseline et avec contexte
    similarities = []
    for i in range(len(baseline_embeddings)):
        sim = cosine_similarity([baseline_embeddings[i]], [context_embeddings[i]])[0][0]
        similarities.append(sim)
    
    # Créer une visualisation des similarités
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(similarities)), similarities)
    
    # Coloration selon l'impact
    for i, sim in enumerate(similarities):
        if sim < 0.7:  # Impact fort
            bars[i].set_color('darkred')
        elif sim < 0.85:  # Impact modéré
            bars[i].set_color('orange')
        else:  # Impact faible
            bars[i].set_color('green')
    
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7)
    plt.ylim(0, 1.01)
    plt.xlabel('Requêtes')
    plt.ylabel('Similarité cosinus (avant/après contexte)')
    plt.title('Impact du contexte mémoriel sur les représentations')
    plt.xticks(range(len(similarities)), query_labels, rotation=45)
    
    plt.tight_layout()
    plt.savefig('memory_effect.png')
    
    # Créer une visualisation PCA des embeddings
    plt.figure(figsize=(12, 8))
    
    # Combiner tous les embeddings pour PCA
    all_embeddings = np.vstack([baseline_embeddings, context_embeddings])
    
    # Réduire la dimensionnalité pour visualisation
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(all_embeddings)
    
    # Séparer les resultats entre baseline et contexte
    n_queries = len(query_labels)
    baseline_reduced = reduced_embeddings[:n_queries]
    context_reduced = reduced_embeddings[n_queries:]
    
    # Tracer les points
    plt.scatter(baseline_reduced[:, 0], baseline_reduced[:, 1], 
                marker='o', color='blue', alpha=0.7, s=100, label='Sans contexte')
    plt.scatter(context_reduced[:, 0], context_reduced[:, 1], 
                marker='x', color='red', alpha=0.7, s=100, label='Avec contexte')
    
    # Connecter les paires baseline-contexte par des flèches
    for i in range(n_queries):
        plt.arrow(baseline_reduced[i, 0], baseline_reduced[i, 1],
                  context_reduced[i, 0] - baseline_reduced[i, 0],
                  context_reduced[i, 1] - baseline_reduced[i, 1],
                  color='gray', alpha=0.5, head_width=0.03, length_includes_head=True)
        
        # Ajouter des labels
        plt.text(baseline_reduced[i, 0], baseline_reduced[i, 1] + 0.05, 
                 query_labels[i], fontsize=9, ha='center')
    
    plt.title('Effet de la mémoire contextuelle sur les représentations')
    plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('memory_effect_pca.png')
    print("Visualisations de l'effet mémoire sauvegardées dans 'memory_effect.png' et 'memory_effect_pca.png'")
    


def main():
    # Créer et configurer le modèle
    config = NeuroLiteConfig.small()
    config.use_external_memory = True
    config.use_dynamic_routing = True
    config.num_experts = 4
    
    print(f"Initialisation du modèle NeuroLite avec mémoire et routage dynamique...")
    model = NeuroLiteModel(config)
    
    # 1. Démonstration du routage dynamique
    print("\n1. Analyse du routage dynamique entre experts")
    print("---------------------------------------------")
    
    # Textes de différentes catégories
    technical_texts = [
        "La programmation orientée objet permet l'encapsulation des données.",
        "Les algorithmes de tri rapide ont une complexité moyenne de O(n log n).",
        "Le protocole HTTPS utilise TLS pour sécuriser les communications."
    ]
    
    # Définir différentes catégories de textes pour tester le routage
    text_categories = {
        "Technique": [
            "Python est un langage de programmation interpreté et orienté objet.",
            "Les réseaux de neurones profonds utilisent des couches de perceptrons.",
            "L'optimisation des hyperparamètres est cruciale en apprentissage automatique."
        ],
        "Littéraire": [
            "La poésie exprime l'âme humaine à travers des mots soigneusement choisis.",
            "Le roman dépeint des personnages complexes dans un univers imaginaire.",
            "L'art de la nouvelle réside dans sa concision et son intensité narrative."
        ],
        "Informel": [
            "Salut, comment ça va aujourd'hui? Tu as des plans pour ce soir?",
            "J'ai trop hâte de voir le dernier film qui vient de sortir!",
            "Franchement, la dernière mise à jour de cette appli est nulle."
        ]
    }
    
    # Aplatir pour l'analyse
    all_texts = []
    text_labels = []
    
    for category, texts in text_categories.items():
        all_texts.extend(texts)
        text_labels.extend([category] * len(texts))
    
    # Analyser les patterns de routage avec le modèle
    print("Analyse des patterns de routage dynamique...")
    activations = analyze_routing_patterns(model, all_texts)
    
    # Générer une heatmap
    if activations is not None and len(activations) > 0:
        plot_routing_heatmap(
            activations, 
            text_labels,
            title="Activation des experts par catégorie de texte"
        )
        
    # 2. Créer un modèle avec capacités AGI
    print("\n=== Création d'un modèle avec capacités AGI ===")
    print("Ce modèle utilise une mémoire hiérarchique multi-niveau")
    
    # Configuration AGI avec mémoire hiérarchique
    agi_config = NeuroLiteConfig.small()
    agi_config.use_external_memory = True
    agi_config.use_dynamic_routing = True
    
    # Utiliser une mémoire hiérarchique au lieu de la mémoire standard
    agi_config.memory_type = "hierarchical"
    agi_config.short_term_size = 64
    agi_config.long_term_size = 128
    agi_config.persistent_size = 256
    
    # Création du modèle AGI
    agi_model = NeuroLiteModel(agi_config)
    
    # 3. Créer un modèle standard pour comparaison
    standard_config = NeuroLiteConfig.small()
    standard_config.use_external_memory = True
    standard_config.memory_type = "standard"
    standard_model = NeuroLiteModel(standard_config)
    
    # 4. Démonstration de la mémoire standard
    print("\n=== Test de mémoire contextuelle standard ===\n")
    
    # Scénario: un sujet cohérent (astrophysique) comme contexte,
    # puis des requêtes liées et non liées
    context_texts = [
        "Les trous noirs sont des régions de l'espace-temps où la gravité est si forte que rien ne peut s'échapper.",
        "Les étoiles à neutrons sont les restes d'une étoile massive après une supernova.",
        "La radiation de Hawking est un rayonnement thermique théorique émis par les trous noirs.",
        "Le Big Bang est le modèle cosmologique dominant décrivant l'origine de l'univers.",
    ]
    
    query_texts = [
        "Que sont les trous noirs supermassifs?",  # Lié directement
        "Comment les étoiles se forment-elles?",  # Lié indirectement
        "Quels sont les objets les plus denses de l'univers?",  # Lié aux étoiles à neutrons
        "Qu'est-ce que la température des galaxies?",  # Partiellement lié
        "Comment fonctionne une voiture électrique?",  # Non lié
    ]
    
    query_labels = [
        "Trous noirs", "Formation stellaire", "Objets denses", "Temp. galaxies", "Voiture élec."
    ]
    
    # Tester l'effet de la mémoire standard
    print("Test avec la mémoire standard...")
    baseline_embs, context_embs = test_memory_retention(standard_model, context_texts, query_texts)
    
    # Visualiser les différences pour le modèle standard
    plot_memory_effect(baseline_embs, context_embs, query_labels)
    
    # 5. Démonstration de la mémoire hiérarchique
    print("\n=== Test de la mémoire hiérarchique (AGI) ===\n")
    
    # Créer une séquence narrative plus longue pour tester la mémoire hiérarchique
    narrative_sequence = [
        # Introduction (mémoire court terme)
        "Notre histoire commence avec les bases de l'astrophysique moderne.",
        "L'univers est vaste et mystérieux, rempli de poussières d'étoiles.",
        "Les galaxies sont les structures fondamentales qui composent le cosmos.",
        "La Voie Lactée est la galaxie qui abrite notre système solaire.",
        
        # Développement (mémoire long terme)
        "Les trous noirs sont parmi les objets les plus fascinants de l'univers.",
        "Ces objets sont si denses que même la lumière ne peut s'échapper de leur attraction.",
        "Les trous noirs supermassifs se trouvent au centre de presque toutes les galaxies.",
        "Ils peuvent contenir la masse de millions, voire de milliards de soleils.",
        "Les étoiles à neutrons sont également des corps célestes extrêmement denses.",
        
        # Conclusion (mémoire persistante)
        "La théorie de la relativité d'Einstein a révolutionné notre compréhension des trous noirs.",
        "L'horizon des événements est la frontière au-delà de laquelle rien ne peut revenir.",
        "La radiation de Hawking suggère que les trous noirs peuvent lentement s'évaporer.",
        "Les observations récentes ont confirmé l'existence des ondes gravitationnelles.",
        "Ces découvertes continuent de redessiner notre vision du cosmos."
    ]
    
    # Tester l'effet de la mémoire hiérarchique
    memory_states = test_hierarchical_memory(agi_model, narrative_sequence, query_texts)
    
    # Visualiser l'évolution à travers les niveaux de mémoire
    plot_hierarchical_memory_effect(memory_states, query_labels)
    
    # 6. Mesurer le temps de réponse
    print("\n=== Comparaison des performances ===\n")
    
    # Générer une séquence test plus longue
    test_sequence = [f"Texte de test numéro {i+1} pour mesurer les performances." for i in range(10)]
    test_query = "Comment est la mémoire de ce modèle?"
    
    # Mesurer le temps pour le modèle standard
    print("Mesure du temps de traitement avec mémoire standard...")
    start_time = time.time()
    with torch.no_grad():
        for text in test_sequence:
            _ = standard_model(input_texts=[text], update_memory=True)
        _ = standard_model(input_texts=[test_query], update_memory=False)
    standard_time = time.time() - start_time
    
    # Mesurer le temps pour le modèle AGI
    print("Mesure du temps de traitement avec mémoire hiérarchique...")
    start_time = time.time()
    with torch.no_grad():
        for text in test_sequence:
            _ = agi_model(input_texts=[text], update_memory=True)
        _ = agi_model(input_texts=[test_query], update_memory=False)
    agi_time = time.time() - start_time
    
    # Afficher les résultats
    print(f"\nTemps d'exécution:")
    print(f"- Modèle standard:    {standard_time:.4f} secondes")
    print(f"- Modèle AGI:        {agi_time:.4f} secondes")
    print(f"- Différence:         {(agi_time-standard_time)/standard_time*100:.1f}%")
    
    print("\nDémonstration terminée. Vérifiez les visualisations générées.")
    print("Pour démontrer la persistance de la mémoire, vous pouvez exécuter ce script à nouveau")
    print("et observer comment le modèle AGI conserve certaines informations entre les exécutions.")

if __name__ == "__main__":
    main()
