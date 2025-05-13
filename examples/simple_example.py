"""
Exemple simple d'utilisation de NeuroLite avec fonctionnalités de base
"""

import torch
import os
from neurolite import NeuroLiteModel, NeuroLiteConfig

def main():
    # Créer une configuration pour un modèle très léger
    config = NeuroLiteConfig.tiny()
    
    # Activer quelques fonctionnalités AGI de base
    config.use_external_memory = True
    config.use_hierarchical_memory = True  # Utiliser la mémoire hiérarchique
    config.short_term_memory_size = 32     # Taille de la mémoire court terme
    config.long_term_memory_size = 64      # Taille de la mémoire long terme
    
    print("Configuration du modèle:")
    print(f"- Taille cachée: {config.hidden_size}")
    print(f"- Nombre de couches: {config.num_mixer_layers}")
    print(f"- Mémoire hiérarchique: activée (Court: {config.short_term_memory_size}, Long: {config.long_term_memory_size})")
    
    # Instancier le modèle
    model = NeuroLiteModel(config)
    
    # Vérifier le nombre de paramètres
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nNombre total de paramètres: {param_count:,}")
    
    # Exemple d'inférence avec des textes bruts
    samples_1 = [
        "NeuroLite est une architecture légère d'IA pour appareils mobiles.",
        "Cette approche combine MLP-Mixer et mémoire associative.",
        "L'objectif est de réduire l'empreinte mémoire tout en maintenant de bonnes performances."
    ]
    
    print("\nTraitement du premier lot de textes:")
    
    # Désactiver le calcul de gradients pour l'inférence
    with torch.no_grad():
        # Premier lot: mise à jour de la mémoire
        outputs_1 = model(input_texts=samples_1, update_memory=True)
        
    # Afficher les dimensions de sortie
    print(f"Forme des sorties: {outputs_1.shape}")
    print(f"- Batch size: {outputs_1.shape[0]}")
    print(f"- Longueur de séquence: {outputs_1.shape[1]}")
    print(f"- Dimension cachée: {outputs_1.shape[2]}")
    
    # Deuxième lot de textes liés au contexte précédent
    samples_2 = [
        "Cette architecture universelle peut évoluer vers des capacités AGI.",
        "Les modèles légers sont essentiels pour l'IA en périphérie.",
        "La mémoire hiérarchique améliore la rétention contextuelle."
    ]
    
    print("\nTraitement du second lot de textes (avec mémoire):")
    
    with torch.no_grad():
        # Second lot: utilise la mémoire du premier lot
        outputs_2 = model(input_texts=samples_2, update_memory=True)
        
        # Traiter à nouveau le premier lot pour démontrer l'effet de la mémoire
        outputs_1_with_memory = model(input_texts=samples_1, update_memory=False)
    
    # Exemple d'utilisation de la représentation (ex: similarité cosinus)
    # Prendre la moyenne sur la dimension de séquence pour obtenir un embedding par texte
    embeddings_1 = torch.mean(outputs_1, dim=1)
    embeddings_2 = torch.mean(outputs_2, dim=1)
    embeddings_1_with_mem = torch.mean(outputs_1_with_memory, dim=1)
    
    # Combiner tous les embeddings pour l'analyse
    all_embeddings = torch.cat([embeddings_1, embeddings_2, embeddings_1_with_mem], dim=0)
    
    # Calculer la similarité cosinus entre toutes les paires
    from torch.nn.functional import normalize
    embeddings_norm = normalize(all_embeddings, p=2, dim=1)
    similarity = torch.mm(embeddings_norm, embeddings_norm.t())
    
    print("\nMatrice de similarité entre les textes:")
    print("(Indices 0-2: premier lot, 3-5: second lot, 6-8: premier lot avec mémoire)")
    print(similarity.numpy().round(2))
    
    # Démontrer l'effet de la mémoire sur les représentations
    print("\nImpact de la mémoire sur les représentations:")
    for i in range(3):
        sim_score = torch.nn.functional.cosine_similarity(
            embeddings_1[i:i+1], embeddings_1_with_mem[i:i+1]
        ).item()
        print(f"Texte {i+1}: Similarité avant/après mémoire = {sim_score:.4f}")
    
    # Sauvegarder le modèle avec sa mémoire persistante
    os.makedirs("models", exist_ok=True)
    model.save_pretrained("models/neurolite-tiny-agi")
    print("\nModèle sauvegardé avec mémoire persistante dans 'models/neurolite-tiny-agi'")
    
    # Charger le modèle sauvegardé (avec sa mémoire)
    print("\nChargement du modèle sauvegardé...")
    loaded_model = NeuroLiteModel.from_pretrained("models/neurolite-tiny-agi")
    
    # Vérifier que le modèle chargé conserve bien sa mémoire
    with torch.no_grad():
        test_output = loaded_model(input_texts=["Test de la mémoire persistante"])
    
    print("Modèle chargé avec succès!\n")

if __name__ == "__main__":
    main()
