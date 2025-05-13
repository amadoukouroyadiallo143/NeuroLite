"""
Exemple d'utilisation de NeuroLite pour une tâche de classification avec fonctionnalités AGI
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from neurolite import NeuroLiteConfig, NeuroLiteForClassification, NeuroLiteModel
from neurolite import HierarchicalMemory, NeurosymbolicReasoner, ContinualAdapter
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

class SimpleTextDataset(Dataset):
    """Dataset simple pour la classification de texte"""
    
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def train_epoch(model, dataloader, optimizer, device):
    """Entraîne le modèle pendant une époque"""
    model.train()
    total_loss = 0
    
    for texts, labels in dataloader:
        # Déplacer les labels vers le device
        labels = labels.to(device)
        
        # Effacer les gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_texts=texts, labels=labels)
        loss = outputs["loss"]
        
        # Backward pass et optimisation
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, silent=False):
    """Évalue le modèle sur un jeu de données"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for texts, labels in dataloader:
            # Déplacer les labels vers le device
            labels = labels.to(device)
            
            # Prédictions du modèle
            outputs = model(input_texts=texts, labels=labels)
            
            # Calculer la perte
            loss = outputs["loss"] if isinstance(outputs, dict) else 0
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            
            # Calculer la précision
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if loss != 0:
                total_loss += loss.item()
    
    # Calculer les métriques finales
    avg_loss = total_loss / len(dataloader) if total_loss > 0 else 0
    accuracy = correct / total
    
    if not silent:
        print(f"Évaluation: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy
    }


def test_continual_learning(model, test_dataset, device):
    """Test de l'apprentissage continu sur des domaines successifs"""
    model.eval()
    
    # Définir deux domaines de textes différents
    domain_a_texts = [
        "L'intelligence artificielle révolutionne la technologie moderne.",
        "Les réseaux de neurones s'inspirent du cerveau humain.",
        "L'apprentissage profond permet de résoudre des problèmes complexes."
    ]
    
    domain_b_texts = [
        "La photosynthèse convertit la lumière solaire en énergie.",
        "Les plantes absorbent le dioxyde de carbone et rejettent de l'oxygène.",
        "Les écosystèmes forestiers abritent une riche biodiversité."
    ]
    
    print("\nTest d'apprentissage continu:")
    print("---------------------------")
    
    # Phase 1: Entraînement sur le domaine A
    print("\nPhase 1: Adaptation au domaine IA...")
    with torch.no_grad():
        for text in domain_a_texts:
            _ = model(input_texts=[text], continuous_learning=True)
    
    # Mesurer la performance sur le jeu de test initial
    print("\nÉvaluation sur le jeu de test initial...")
    accuracy_a1 = evaluate(model, DataLoader(
        test_dataset, batch_size=16, shuffle=False
    ), device, silent=True)["accuracy"]
    
    # Phase 2: Adaptation au domaine B
    print("\nPhase 2: Adaptation au domaine biologie...")
    with torch.no_grad():
        for text in domain_b_texts:
            _ = model(input_texts=[text], continuous_learning=True)
    
    # Re-tester sur le domaine initial
    print("\nRéévaluation sur le jeu de test initial...")
    accuracy_a2 = evaluate(model, DataLoader(
        test_dataset, batch_size=16, shuffle=False
    ), device, silent=True)["accuracy"]
    
    # Calculer la différence
    forgetting = accuracy_a1 - accuracy_a2
    
    print(f"\nRésultats d'apprentissage continu:")
    print(f"- Précision initiale: {accuracy_a1:.4f}")
    print(f"- Précision après adaptation: {accuracy_a2:.4f}")
    print(f"- Oubli catastrophique: {forgetting:.4f} ({forgetting*100:.1f}%)")
    
    if forgetting < 0.05:
        print("\n✓ Excellent: Très peu d'oubli catastrophique détecté!")
    elif forgetting < 0.15:
        print("\n△ Acceptable: Niveau d'oubli modéré")
    else:
        print("\n✗ Attention: Niveau d'oubli catastrophique significatif")


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du dispositif: {device}")
    
    # Créer un dataset simple pour les démonstrations
    num_samples = 1000
    num_classes = 5
    
    # Générer des données synthétiques
    print("\nCréation d'un dataset synthétique...")
    texts = []
    labels = []
    
    topics = [
        "intelligence artificielle",
        "apprentissage profond",
        "traitement du langage naturel",
        "vision par ordinateur",
        "robotique"
    ]
    
    templates = [
        "Cet article traite de {topic} et de ses applications.",
        "Une analyse récente sur {topic} montre des avancées significatives.",
        "Les dernières recherches en {topic} sont prometteuses.",
        "Comment {topic} transforme l'industrie moderne.",
        "L'évolution de {topic} ces dernières années est remarquable."
    ]
    
    # Générer des textes et étiquettes
    np.random.seed(42)  # Pour reproductibilité
    for _ in range(num_samples):
        class_idx = np.random.randint(0, num_classes)
        topic = topics[class_idx]
        template = np.random.choice(templates)
        text = template.format(topic=topic)
        
        texts.append(text)
        labels.append(class_idx)
    
    # Créer le dataset
    dataset = SimpleTextDataset(texts, labels)
    
    # Diviser en train/test (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Créer les dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Créer le modèle
    config = NeuroLiteConfig.tiny()
    # Créer deux configurations de modèle
    print("\nInitialisation des modèles...")
    
    # Configuration de base
    standard_config = NeuroLiteConfig.tiny()
    standard_config.num_classes = num_classes
    
    # Configuration AGI avec fonctionalités avancées
    agi_config = NeuroLiteConfig.tiny()
    agi_config.num_classes = num_classes
    agi_config.use_hierarchical_memory = True
    agi_config.use_external_memory = True
    agi_config.use_advanced_reasoning = True
    agi_config.use_continual_learning = True
    
    # Créer les modèles
    standard_model = NeuroLiteForClassification(standard_config)
    standard_model.to(device)
    
    agi_model = NeuroLiteForClassification(agi_config)
    agi_model.to(device)
    
    # Nombres de paramètres
    standard_param_count = sum(p.numel() for p in standard_model.parameters() if p.requires_grad)
    agi_param_count = sum(p.numel() for p in agi_model.parameters() if p.requires_grad)
    
    print(f"NeuroLite standard: {standard_param_count:,} paramètres")
    print(f"NeuroLite AGI:      {agi_param_count:,} paramètres (+{agi_param_count-standard_param_count:,})")
    
    # Définir les hyperparamètres
    batch_size = 16
    epochs = 5
    learning_rate = 1e-3
    
    # Définir l'optimiseur
    optimizer = optim.Adam(agi_model.parameters(), lr=learning_rate)
    
    # Entraînement des modèles
    print("\nEntraînement des modèles...")
    num_epochs = 3
    batch_size = 32
    
    optimizer_standard = optim.Adam(standard_model.parameters(), lr=0.001)
    optimizer_agi = optim.Adam(agi_model.parameters(), lr=0.001)
    
    print("\n1. Entraînement du modèle standard:")
    for epoch in range(num_epochs):
        train_loss = train_epoch(standard_model, train_loader, optimizer_standard, device)
        val_metrics = evaluate(standard_model, test_loader, device)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss: {val_metrics['loss']:.4f}")
        print(f"  Test Accuracy: {val_metrics['accuracy']:.4f}")
    
    print("\n2. Entraînement du modèle avec fonctionnalités AGI:")
    for epoch in range(num_epochs):
        train_loss = train_epoch(agi_model, train_loader, optimizer_agi, device)
        val_metrics = evaluate(agi_model, test_loader, device)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss: {val_metrics['loss']:.4f}")
        print(f"  Test Accuracy: {val_metrics['accuracy']:.4f}")
    
    # Évaluation sur le jeu de test
    print("\nÉvaluation finale sur le jeu de test...")
    
    print("\n1. Modèle standard:")
    standard_test_metrics = evaluate(standard_model, test_loader, device)
    print(f"Test Loss: {standard_test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {standard_test_metrics['accuracy']:.4f}")
    
    print("\n2. Modèle avec fonctionnalités AGI:")
    agi_test_metrics = evaluate(agi_model, test_loader, device)
    print(f"Test Loss: {agi_test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {agi_test_metrics['accuracy']:.4f}")
    
    # Rapport de classification détaillé pour le modèle AGI
    y_true = []
    y_pred = []
    
    agi_model.eval()
    
    # Sauvegarder le modèle
    import os
    os.makedirs("models", exist_ok=True)
    model.save_pretrained("models/neurolite-sentiment")
    print("Modèle sauvegardé dans 'models/neurolite-sentiment'")
    
    # Démonstration d'inférence
    print("\nDémonstration d'inférence:")
    test_samples = [
        "Ce produit a dépassé toutes mes attentes, un vrai plaisir à utiliser !",
        "Complètement déçu par cet achat, je ne recommande pas."
    ]
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_texts=test_samples)
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        for text, pred in zip(test_samples, preds):
            sentiment = "positif" if pred == 1 else "négatif"
            print(f"Texte: {text}")
            print(f"Sentiment prédit: {sentiment}\n")

    # Visualisation et comparaison des embeddings
    print("\nGénération des embeddings pour visualisation...")
    
    # Obtenir les embeddings pour un sous-ensemble de données
    num_vis_samples = 200
    vis_texts = []
    vis_labels = []
    vis_embeddings_standard = []
    vis_embeddings_agi = []
    
    with torch.no_grad():
        for i in range(min(num_vis_samples, len(test_dataset))):
            text, label = test_dataset[i]
            # Embeddings modèle standard
            standard_model.eval()
            outputs_standard = standard_model(input_texts=[text])
            embedding_standard = torch.mean(outputs_standard, dim=1).cpu().numpy()
            
            # Embeddings modèle AGI
            agi_model.eval()
            outputs_agi = agi_model(input_texts=[text])
            embedding_agi = torch.mean(outputs_agi, dim=1).cpu().numpy()
            
            vis_texts.append(text)
            vis_labels.append(label)
            vis_embeddings_standard.append(embedding_standard[0])
            vis_embeddings_agi.append(embedding_agi[0])
    
    # Réduction de dimension avec t-SNE
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    
    # Embeddings du modèle standard
    vis_embeddings_standard = np.array(vis_embeddings_standard)
    reduced_embeddings_standard = tsne.fit_transform(vis_embeddings_standard)
    
    # Embeddings du modèle AGI
    vis_embeddings_agi = np.array(vis_embeddings_agi)
    reduced_embeddings_agi = tsne.fit_transform(vis_embeddings_agi)
    
    # Créer une figure comparative
    plt.figure(figsize=(16, 8))
    
    # Graphique pour le modèle standard
    plt.subplot(1, 2, 1)
    for i in range(num_classes):
        indices = [j for j, label in enumerate(vis_labels) if label == i]
        plt.scatter(
            reduced_embeddings_standard[indices, 0],
            reduced_embeddings_standard[indices, 1],
            label=topics[i],
            alpha=0.7
        )
    plt.title("Embeddings du modèle standard")
    plt.legend()
    
    # Graphique pour le modèle AGI
    plt.subplot(1, 2, 2)
    for i in range(num_classes):
        indices = [j for j, label in enumerate(vis_labels) if label == i]
        plt.scatter(
            reduced_embeddings_agi[indices, 0],
            reduced_embeddings_agi[indices, 1],
            label=topics[i],
            alpha=0.7
        )
    plt.title("Embeddings du modèle avec fonctionnalités AGI")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("embeddings_comparison.png")
    print("Visualisation comparative sauvegardée dans 'embeddings_comparison.png'")
    
    # Calcul de la cohérence des clusters
    from sklearn.metrics import silhouette_score
    
    try:
        silhouette_standard = silhouette_score(vis_embeddings_standard, vis_labels)
        silhouette_agi = silhouette_score(vis_embeddings_agi, vis_labels)
        
        print("\nCohérence des clusters (score silhouette):")
        print(f"- Modèle standard: {silhouette_standard:.4f}")
        print(f"- Modèle AGI:      {silhouette_agi:.4f}")
        
        if silhouette_agi > silhouette_standard:
            print("✓ Les fonctionnalités AGI améliorent la séparation des classes!")
        else:
            print("Les fonctionnalités AGI n'améliorent pas significativement la séparation des classes.")
    except Exception as e:
        print(f"Impossible de calculer les scores silhouette: {e}")


if __name__ == "__main__":
    main()
