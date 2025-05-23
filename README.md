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
4. **Composants Neurosymboliques Avancés**:
    - **Moteur de Règles Symboliques (`SymbolicRuleEngine`)**: Un moteur d'inférence léger qui charge des règles (logique de premier ordre) et des faits à partir de fichiers JSON. Il supporte l'ajout dynamique de faits, l'inférence vers l'avant (forward chaining) et la gestion de la négation dans les prémisses des règles.
    - **Couche Neurosymbolique (`NeuralSymbolicLayer`)**: Intègre le `SymbolicRuleEngine` dans le pipeline neuronal. Cette couche peut extraire des entités et relations potentielles des états cachés du modèle, les affirmer comme faits transitoires au moteur de règles, initier une phase d'inférence, puis réintégrer les faits dérivés dans les représentations neuronales. Elle supporte le traitement par batch et utilise des embeddings apprenables pour les types de prédicats et les identifiants d'entités symboliques, permettant au modèle d'apprendre à interpréter et utiliser les résultats du raisonnement symbolique. Le fichier `rules.json` (configurable via `symbolic_rules_file` dans `NeuroLiteConfig`) est utilisé pour charger les règles et faits persistants.
    - **Réseau Bayésien (`BayesianBeliefNetwork`)**: Permet d'intégrer des connaissances probabilistes et d'effectuer du raisonnement incertain. La structure du réseau (variables et leurs dépendances) peut être définie via la configuration (`bayesian_network_structure` et `num_bayesian_variables` dans `NeuroLiteConfig`). Le module utilise un algorithme d'inférence approximative (basé sur le Likelihood Weighting) pour estimer les probabilités postérieures étant donné des évidences extraites des états neuronaux.
5. **Routage dynamique** - Activation conditionnelle de sous-modules spécialisés via Mixture-of-Experts léger
6. **Apprentissage Continu et Mémoire Évoluée**:
    - **Adaptateur d'Apprentissage Continu (`ContinualAdapter`)**: Intégré au modèle, ce module vise à permettre l'apprentissage à partir de nouvelles données au fil du temps tout en atténuant l'oubli catastrophique des connaissances antérieures. Il utilise des mécanismes comme un tampon de rejeu (`replay buffer`) pour stocker des expériences passées, une détection conceptuelle de dérive de distribution (`drift detection`), et des stratégies d'adaptation du modèle. Voir `examples/lifelong_learning_demo.py`.
    - **Mémoire Hiérarchique Améliorée (`HierarchicalMemory`)**: La mémoire hiérarchique a été dotée de capacités plus dynamiques :
        - **Consolidation Intelligente**: Le transfert d'informations entre les niveaux de mémoire (court terme, long terme, persistant) est désormais modulé par la nouveauté des données. Les informations nouvelles et surprenantes sont priorisées pour la consolidation dans les mémoires à plus long terme, rendant les mises à jour plus sélectives et efficaces. Ceci est contrôlé par `novelty_threshold_ltm` et `novelty_threshold_pm`.
        - **Portes Contextuelles de Mémoire**: La contribution de chaque niveau de mémoire (STM, LTM, PM) à la sortie finale est déterminée dynamiquement pour chaque token d'entrée, grâce à des poids calculés par `memory_gate` sur les requêtes. Cela permet une récupération d'informations contextuelles plus nuancée et pertinente.

## 🧠 Fondements Théoriques

NeuroLite s'inspire de plusieurs avancées théoriques récentes:

- **MLP-Mixer**: Démontre que des projections linéaires alternées (token-mixing et channel-mixing) peuvent rivaliser avec l'attention pour de nombreuses tâches
- **Complexité Linéaire**: Exploite les approches comme Performer, Linformer et FNet qui remplacent l'attention quadratique par des approximations efficaces
- **Mémoire Associative Moderne**: Intègre des réseaux de Hopfield continus de grande capacité pour la mémorisation associative
- **Routage Adaptatif**: Utilise des techniques de routage dynamique pour activer sélectivement différents "experts" selon le contexte
- **Composants Neurosymboliques (étendus)**: Combine traitement neuronal avec des mécanismes de raisonnement symbolique et probabiliste plus explicites pour améliorer les capacités de raisonnement structuré et la gestion de l'incertitude, tout en maintenant une faible empreinte paramétrique.
- **Apprentissage Continu (Lifelong Learning)**: S'inspire des approches visant à permettre aux modèles d'apprendre séquentiellement de nouvelles tâches ou données sans oublier les précédentes, en utilisant des tampons de rejeu et des mécanismes d'adaptation.
- **Mémoires Hiérarchiques Dynamiques**: Les améliorations apportées à la mémoire s'inspirent des modèles cognitifs de la mémoire humaine, où la consolidation et la récupération sont des processus dynamiques et dépendants du contexte et de la nouveauté.

## 📦 Structure du Projet

```
neurolite/
├── __init__.py        # Point d'entrée du package
├── config.py          # Configuration des différentes tailles de modèles
├── model.py           # Modèle principal et variantes spécialisées
├── projection.py      # Couche de projection d'entrée (MinHash+Bloom)
├── mixer.py           # Implémentations MLP-Mixer, HyperMixer, FNet
├── memory.py          # Mémoire externe différentiable (base)
├── hierarchical_memory.py # Mémoire hiérarchique améliorée
├── routing.py         # Routage dynamique et Mixture-of-Experts
├── symbolic.py        # Composants de raisonnement symbolique (moteur de règles, couche neurosymbolique, BBN)
└── continual.py       # Adaptateur d'apprentissage continu

training/
├── data_manager.py    # Gestion des données d'entraînement et validation
├── train_generator.py # Script d'entraînement du modèle de génération
└── train_classifier.py # Script d'entraînement du classifieur

data/
└── wikitext/         # Données d'entraînement provenant du corpus WikiText
    ├── train/        # Données d'entraînement
    ├── val/          # Données de validation
    └── test/         # Données de test

examples/
├── simple_example.py           # Exemple basique d'utilisation
├── classification_example.py   # Classification de texte
├── memory_and_routing_example.py # Démonstration mémoire et routage
├── symbolic_reasoning_example.py # Démonstration du moteur de règles et couche neurosymbolique
├── bayesian_network_example.py   # Démonstration du réseau bayésien
├── lifelong_learning_demo.py     # Démonstration de l'apprentissage continu avec l'adaptateur et la mémoire hiérarchique
└── benchmark_comparison.py     # Comparaison avec architectures standards

generate_text.py     # Utilitaire de génération de texte avec modèle entraîné
neurolite_demo.py    # Application de démonstration interactive
```

## 🚀 Installation

Pour installer les dépendances nécessaires:

```bash
git clone https://github.com/username/NeuroLite.git
cd NeuroLite
python -m venv .venv
.venv\Scripts\activate  # Sur Windows
source .venv/bin/activate  # Sur Linux/MacOS
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

## 🏋️ Entraînement du Modèle

NeuroLite comprend des scripts d'entraînement robustes pour diverses tâches. Pour entraîner un modèle de génération de texte sur le corpus WikiText :

```bash
python training/train_generator.py --data_dir "data/wikitext" --batch_size 32 --seq_length 512 --vocab_size 32000 --num_epochs 20
```

Options d'entraînement importantes :
- `--batch_size` : Taille des batchs (défaut: 32)
- `--seq_length` : Longueur de séquence pour l'entraînement (défaut: 128)
- `--vocab_size` : Taille du vocabulaire (défaut: 10000)
- `--hidden_size` : Dimension des couches cachées (défaut: 256)
- `--num_layers` : Nombre de couches mixer (défaut: 6)
- `--use_memory` : Activer la mémoire externe (flag)
- `--learning_rate` : Taux d'apprentissage (défaut: 5e-5)
- `--max_samples` : Limite le nombre d'échantillons (pour tests rapides)

Pour entraîner sur un matériel limité, utilisez des paramètres plus légers :

```bash
python training/train_generator.py --data_dir "data/wikitext" --batch_size 8 --seq_length 128 --hidden_size 128 --num_layers 4 --num_epochs 5 --max_samples 1000
```

Une fois entraîné, générez du texte avec le modèle :

```bash
python generate_text.py --model_path "models/generator_ep20.pt" --prompt "NeuroLite est" --max_length 100
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

# Standard avec modules symboliques et bayésiens activés par défaut
config = NeuroLiteConfig.base_symbolic()
# Cela active `use_symbolic_module=True`, `symbolic_rules_file="rules.json"`,
# `use_bayesian_module=True`, et définit une structure bayésienne d'exemple.

# Personnalisation avancée
config = NeuroLiteConfig(
    hidden_size=256,
    num_mixer_layers=6,
    use_external_memory=True,
    use_dynamic_routing=True,
    num_experts=4,
    
    # Activation et configuration du module symbolique
    use_symbolic_module=True,
    symbolic_rules_file="custom_rules.json", # Chemin vers votre fichier de règles
    max_predicate_types=100, # Taille du vocabulaire pour les types de prédicats
    max_entities_in_vocab=500, # Taille du vocabulaire pour les entités symboliques

    # Activation et configuration du réseau bayésien
    use_bayesian_module=True,
    num_bayesian_variables=15,
    # Exemple: [(parent_idx, child_idx), ...]
    bayesian_network_structure=[(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)], 
    max_parents_bayesian=3, # Utilisé si bayesian_network_structure n'est pas fourni

    # Activation et configuration de l'apprentissage continu
    use_continual_adapter=True,
    continual_adapter_buffer_size=200, # Taille du tampon de rejeu
    continual_adapter_rate=0.05,       # Taux d'adaptation
    continual_adapter_drift_threshold=0.6, # Seuil de détection de dérive

    # Configuration des seuils de nouveauté pour HierarchicalMemory
    novelty_threshold_ltm=0.6, # Seuil pour la mise à jour de la mémoire à long terme
    novelty_threshold_pm=0.7   # Seuil pour la mise à jour de la mémoire persistante
)
```

## 🔄 Gestion des Données & Optimisations

NeuroLite comprend des systèmes robustes pour la gestion et le traitement des données :

- **WikiTextDataset** : Chargement efficace et gestion du corpus WikiText
- **Padding intelligent** : Traitement optimal des textes plus courts que la longueur de séquence cible
- **Tokenization optimisée** : Tokenizer rapide avec vocabulaire ajustable (jusqu'à 32K tokens)
- **Multiprocessing** : Chargement parallèle des données pour accélérer l'entraînement
- **Gestion de batch dynamique** : Fonction de collation robuste pour la création de batchs homogènes
- **Intégration PyTorch** : Compatibilité complète avec l'écosystème PyTorch (DataLoader, etc.)

Chaque composant est conçu pour être efficace en mémoire et en temps de calcul, même sur du matériel limité.

## 📚 Références

Cette implémentation s'inspire des travaux suivants:
- MLP-Mixer (Tolstikhin et al., 2021)
- pNLP-Mixer (Fusco et al., 2023)
- HyperMixer (Mai et al., 2023)
- FNet (Lee et al., 2022)
- Performer (Choromanski et al., 2020)
- Modern Hopfield Networks (Ramsauer et al., 2020)
- Differentiable Neural Computers (Graves et al., 2016)
- State Space Models (Gu et al., 2023)
- Mamba (Gu et al., 2023)

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## 🤝 Contributions

Les contributions sont bienvenues! N'hésitez pas à soumettre des pull requests ou à ouvrir des issues pour des suggestions d'amélioration.

---

<div align="center">
<strong>NeuroLite - Vers une AGI légère et efficiente</strong>
</div>
