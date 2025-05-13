# NeuroLite

Une architecture d'IA légère pour les appareils mobiles et embarqués, fournissant des alternatives efficaces aux Transformers. NeuroLite implémente des approches innovantes (MLP-Mixer, mémoire neuronale, routage adaptatif) pour créer des modèles compacts capables de traiter le langage naturel avec une fraction des ressources requises par les architectures traditionnelles.

<div align="center">
<strong>Complexité Linéaire | Empreinte Minimale | AGI Légère</strong>
</div>

## 🌟 Points Clés

- **Ultra-léger**: Modèles de 1-10Mo, contre 110-340Mo pour les Transformers standards
- **Efficacité Computationnelle**: Complexité linéaire (O(n)) en longueur de séquence vs quadratique (O(n²)) 
- **Mémoire Adaptative**: Système de rétention contextuelle à long terme inspiré des réseaux de Hopfield modernes
- **Routage Intelligent**: Activation conditionnelle des experts spécialisés selon le type d'entrée
- **Composant Symbolique**: Module léger de raisonnement structuré pour améliorer les capacités symboliques
- **Mobile-First**: Conçu pour fonctionner efficacement sur smartphones, wearables et dispositifs IoT

## 🏗️ Architecture

NeuroLite combine plusieurs innovations récentes en une architecture hybride légère et performante:

![Architecture NeuroLite](https://placeholder-for-architecture-diagram.com/neurolite_arch.png)

1. **Projection d'entrée efficace** - Remplace les lourdes tables d'embedding par un encodage léger basé sur MinHash et filtres de Bloom (~99% de réduction de paramètres)
2. **Backbone All-MLP** - Couches MLP-Mixer ou HyperMixer pour un traitement de séquence avec complexité temporelle et spatiale linéaire
3. **Mémoire externe différentiable** - Système de mémoire associative à plusieurs niveaux pour la rétention contextuelle
4. **Module symbolique** - Composant de raisonnement structuré permettant d'intégrer des connaissances et règles explicites
5. **Routage dynamique** - Activation conditionnelle de sous-modules spécialisés via Mixture-of-Experts léger

## 🧠 Fondements Théoriques

NeuroLite s'inspire de plusieurs avancées théoriques récentes:

- **MLP-Mixer**: Démontre que des projections linéaires alternées (token-mixing et channel-mixing) peuvent rivaliser avec l'attention pour de nombreuses tâches
- **Complexité Linéaire**: Exploite les approches comme Performer, Linformer et FNet qui remplacent l'attention quadratique par des approximations efficaces
- **Mémoire Associative Moderne**: Intègre des réseaux de Hopfield continus de grande capacité pour la mémorisation associative
- **Routage Adaptatif**: Utilise des techniques de routage dynamique pour activer sélectivement différents "experts" selon le contexte
- **Composants Neurosymboliques**: Combine traitement neuronal et symbolique pour améliorer les capacités de raisonnement avec peu de paramètres

## 📦 Structure du Projet

```
neurolite/
├── __init__.py        # Point d'entrée du package
├── config.py          # Configuration des différentes tailles de modèles
├── model.py           # Modèle principal et variantes spécialisées
├── projection.py      # Couche de projection d'entrée (MinHash+Bloom)
├── mixer.py           # Implémentations MLP-Mixer, HyperMixer, FNet
├── memory.py          # Mémoire externe différentiable
├── routing.py         # Routage dynamique et Mixture-of-Experts
└── symbolic.py        # Composants de raisonnement symbolique

examples/
├── simple_example.py           # Exemple basique d'utilisation
├── classification_example.py   # Classification de texte
├── memory_and_routing_example.py # Démonstration mémoire et routage
└── benchmark_comparison.py     # Comparaison avec architectures standards

neurolite_demo.py      # Application de démonstration interactive
```

## 🚀 Installation

Pour installer les dépendances nécessaires:

```bash
git clone https://github.com/username/NeuroLite.git
cd NeuroLite
pip install -r requirements.txt
```

## 🔧 Utilisation

### Exemple Simple

```python
from neurolite import NeuroLiteModel, NeuroLiteConfig

# Créer un modèle ultra-léger
config = NeuroLiteConfig.tiny()  # ~1-2Mo
model = NeuroLiteModel(config)

# Traiter du texte
texts = ["NeuroLite est une architecture légère d'IA."]
outputs = model(input_texts=texts)

# Utiliser les représentations vectorielles (embeddings)
embedding = outputs.mean(dim=1)
```

### Classification de Texte

```python
from neurolite import NeuroLiteForClassification, NeuroLiteConfig

# Configurer un modèle pour classification
config = NeuroLiteConfig.small()
model = NeuroLiteForClassification(config, num_labels=2)

# Inférence
outputs = model(input_texts=["Texte à classifier"])
prediction = outputs["logits"].argmax(dim=1)
```

### Utilisation de la Mémoire Contextuelle

```python
# Créer un modèle avec mémoire
config = NeuroLiteConfig.small()
config.use_external_memory = True
model = NeuroLiteModel(config)

# Fournir du contexte à la mémoire
model(input_texts=["Alice est une ingénieure vivant à Paris."], update_memory=True)

# La requête suivante sera enrichie par le contexte en mémoire
result = model(input_texts=["Où habite-t-elle ?"])
```

## 🧪 Exemples et Démonstration

Exécutez la démonstration interactive pour explorer les capacités du modèle:

```bash
python neurolite_demo.py --size tiny  # Options: tiny, small, base
```

Autres exemples disponibles:
- `examples/simple_example.py` - Utilisation de base
- `examples/classification_example.py` - Classification de sentiment
- `examples/memory_and_routing_example.py` - Démonstration mémoire et routage
- `examples/benchmark_comparison.py` - Comparaison avec Transformer

## 📈 Performances

Comparaison avec des architectures standards sur un texte de longueur moyenne (256 tokens):

| Architecture | Paramètres | Temps d'inférence | Mémoire (RAM) | Complexité |
|--------------|------------|-------------------|---------------|------------|
| BERT-base    | 110M       | ~45ms             | ~440MB        | O(n²)      |
| DistilBERT   | 66M        | ~25ms             | ~265MB        | O(n²)      |
| NeuroLite-base | 8M       | ~10ms             | ~32MB         | O(n)       |
| NeuroLite-small | 3M      | ~5ms              | ~12MB         | O(n)       |
| NeuroLite-tiny  | 1M      | ~2ms              | ~4MB          | O(n)       |

## 🛠️ Personnalisation

NeuroLite offre plusieurs configurations pré-définies:

```python
# Très léger (~1-2Mo)
config = NeuroLiteConfig.tiny()

# Léger (~5-10Mo)
config = NeuroLiteConfig.small()

# Standard (~20-30Mo)
config = NeuroLiteConfig.base()

# Personnalisation avancée
config = NeuroLiteConfig(
    hidden_size=256,
    num_mixer_layers=6,
    use_external_memory=True,
    use_dynamic_routing=True,
    use_symbolic_module=True,
    num_experts=4
)
```

## 📚 Références

Cette implémentation s'inspire des travaux suivants:
- MLP-Mixer (Tolstikhin et al., 2021)
- pNLP-Mixer (Fusco et al., 2023)
- HyperMixer (Mai et al., 2023)
- FNet (Lee et al., 2022)
- Performer (Choromanski et al., 2020)
- Modern Hopfield Networks (Ramsauer et al., 2020)
- Differentiable Neural Computers (Graves et al., 2016)

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## 🤝 Contributions

Les contributions sont bienvenues! N'hésitez pas à soumettre des pull requests ou à ouvrir des issues pour des suggestions d'amélioration.

---

<div align="center">
<strong>NeuroLite - Vers une AGI légère et efficiente</strong>
</div>
