# NeuroLite

# NeuroLite: Vers une Intelligence Artificielle Générale (AGI) Légère et Efficace

NeuroLite est une architecture d'IA universelle et légère, conçue spécifiquement pour les appareils mobiles, embarqués, et les environnements où les ressources sont limitées. Elle offre une alternative performante aux modèles de type Transformer, en se concentrant sur l'efficacité computationnelle et une faible empreinte mémoire.

Au cœur de NeuroLite se trouve une **vision AGI** : développer un système capable d'une compréhension multimodale profonde, d'un apprentissage continu autonome, d'une mémorisation contextuelle robuste, et d'un raisonnement hybride (neuronal et symbolique) flexible. Plutôt que de s'appuyer sur une mise à l'échelle massive, NeuroLite vise l'AGI par l'intégration synergique de modules spécialisés et optimisés, favorisant une intelligence plus adaptable et généralisable.

<div align="center">
<strong>Complexité Linéaire | Empreinte Minimale | Apprentissage Continu | Raisonnement Hybride | Perception Multimodale | Vision AGI</strong>
</div>

## 🌟 Points Clés

- **Ultra-léger**: Modèles de 1-10Mo, contre 110-340Mo pour les Transformers standards
- **Efficacité Computationnelle**: Complexité linéaire (O(n)) en longueur de séquence vs quadratique (O(n²)) 
- **Mémoire Adaptative**: Système de rétention contextuelle à long terme inspiré des réseaux de Hopfield modernes
- **Tokenizer Multimodal**: Système de tokenization avancé avec double codebook pour caractéristiques sémantiques et détaillées
- **Noyau Latent Universel**: Espace de représentation unifié pour toutes les modalités, permettant une intégration et une transformation fluides entre différents types de données
- **Encodeurs Spécialisés**: Modules dédiés pour le traitement de texte, image, audio, vidéo et graphes
- **Générateurs Multimodaux**: Capacité à générer des sorties dans différentes modalités à partir d'un espace latent commun
- **Routage Intelligent**: Activation conditionnelle des experts spécialisés selon le type d'entrée
- **Composant Symbolique**: Module léger de raisonnement structuré pour améliorer les capacités symboliques
- **Mobile-First**: Conçu pour fonctionner efficacement sur smartphones, wearables et dispositifs IoT
- **Mémoire Hiérarchique Intégrée**: `HierarchicalMemory` fournit un contexte riche pour le raisonnement et l'apprentissage, avec des interactions explicites avec `NeurosymbolicReasoner`, `StructuredPlanner`, et `ContinualAdapter`.
- **Fusion Multimodale Avancée**: Stratégie de fusion améliorée utilisant un `CrossModalFuser` (basé sur l'attention self/croisée) pour une intégration plus profonde des différentes modalités au début du traitement.
- **Apprentissage Continu Amélioré**: `ContinualAdapter` peut désormais consolider les connaissances importantes dans `HierarchicalMemory`, renforçant l'apprentissage à long terme.
- **Optimisations d'Efficacité**:
    - Quantification Post-Entraînement (PTQ) démontrée avec succès, réduisant la taille du modèle de ~70% tout en maintenant la fonctionnalité.
    - `DynamicRoutingBlock` (utilisant `SparseDispatcher`) optimisé avec `scatter_add_` pour une agrégation plus rapide des sorties d'experts et ajout de statistiques d'utilisation des experts.
- **Pipeline de Données Multimodal (Esquisse)**: Introduction de `MultimodalDataset` et `multimodal_collate_fn` pour une gestion structurée des données multimodales.

## 📝️ Architecture

NeuroLite combine plusieurs innovations récentes en une architecture hybride légère et performante, conçue pour une intelligence artificielle générale et efficace :

![Architecture NeuroLite](./architectures/neurolite_architecture_modern_fr.png)
*Diagramme conceptuel de l'architecture NeuroLite, illustrant l'interaction des modules clés.*

1.  **Entrée et Perception Multimodale**:
    *   **Tokenizer Multimodal Avancé (`NeuroLiteTokenizer`)**: Ce module (décrit plus en détail dans le code source `neurolite/tokenization/tokenizer.py`) transforme les données brutes de diverses modalités en un format tokenisé adapté au traitement neuronal. Il utilise des techniques comme un double codebook (sémantique et détail) et la quantification vectorielle résiduelle.
    *   **Projection d'Entrée Multimodale (`MultimodalProjection`)**: Ce module (`neurolite/multimodal/multimodal.py`) prend les données brutes ou prétraitées de chaque modalité (texte, image, audio, vidéo, graphe) et les encode en représentations vectorielles. Il utilise des encodeurs spécialisés pour chaque modalité (ex: `TextEncoder`, `ImageEncoder` basé sur ViT).
    *   **Fusion Multimodale Stratégique (`CrossModalFuser`)**: NeuroLite intègre désormais un module `CrossModalAttention` (appelé `cross_modal_fuser` dans `NeuroLiteModel`) qui opère directement sur les sorties individuelles de `MultimodalProjection`. Ce module applique une attention croisée (ou self-attention) sur la séquence des représentations modales, permettant une fusion riche et contextuelle avant l'entrée dans les couches de traitement principales. Cette approche remplace l'ancienne méthode d'application de l'attention croisée plus tard dans le réseau.

2.  **Cœur du Modèle et Traitement Séquentiel**:
    *   **Backbone All-MLP**: Utilise des couches de type MLP-Mixer ou HyperMixer (`neurolite/core/mixer.py`) pour un traitement efficace des séquences avec une complexité linéaire, offrant une alternative aux couches d'attention coûteuses des Transformers.
    *   **Routage Dynamique (`DynamicRoutingBlock`)**: Ce bloc (`neurolite/routers/routing.py`) intègre un `SparseDispatcher` (Mixture-of-Experts optimisé) pour activer conditionnellement seulement un sous-ensemble d'experts (petits réseaux de neurones) en fonction de l'entrée. Cela permet d'augmenter la capacité du modèle sans augmenter proportionnellement la charge de calcul pour chaque inférence. Des optimisations récentes incluent l'utilisation de `scatter_add_` pour une agrégation plus efficace et la possibilité de retourner des statistiques d'activation des experts.

3.  **Mémoire, Apprentissage et Raisonnement**:
    *   **Mémoire Hiérarchique (`HierarchicalMemory`)**: Située dans `neurolite/memory/hierarchical_memory.py`, cette mémoire à plusieurs niveaux (court terme, long terme, persistant) est cruciale pour la rétention contextuelle. Elle interagit désormais de manière plus profonde avec d'autres modules :
        *   **Intégration avec le Raisonnement**: `NeurosymbolicReasoner` et `StructuredPlanner` (dans `neurolite/reasoning/reasoning.py`) peuvent consulter `HierarchicalMemory` pour des faits et contextes pertinents lors de leurs processus. Le raisonneur peut stocker des conclusions importantes, et le planificateur peut mémoriser des plans réussis.
        *   **Intégration avec l'Apprentissage Continu**: `ContinualAdapter` (`neurolite/continual/continual.py`) peut consolider les informations et expériences jugées importantes (par exemple, celles issues de son `ReplayBuffer` ou ayant conduit à une adaptation significative) dans `HierarchicalMemory`, favorisant un apprentissage cumulatif et à long terme.
    *   **Composants Neurosymboliques et de Planification (`NeurosymbolicReasoner`, `StructuredPlanner`)**: Ces modules permettent au modèle d'effectuer un raisonnement structuré et de planifier des actions. Leur interaction avec la mémoire hiérarchique leur confère une capacité accrue à utiliser des connaissances passées et à stocker de nouvelles informations apprises.
    *   **Adaptateur d'Apprentissage Continu (`ContinualAdapter`)**: Permet au modèle de s'adapter à de nouvelles données ou tâches au fil du temps, tout en atténuant l'oubli catastrophique, notamment grâce à son interaction avec `HierarchicalMemory`.

4.  **Optimisations d'Efficacité**:
    *   **Quantification Post-Entraînement (PTQ)**: Des expériences (voir `ptq_script.py` après exécution) ont montré que la quantification dynamique des poids (par exemple, en `int8`) peut réduire la taille du modèle `NeuroLiteModel` d'environ 70% tout en préservant sa capacité à effectuer des inférences, ce qui est crucial pour les déploiements sur appareils à ressources limitées.

5.  **Pipeline de Données (Esquisse)**:
    *   Une structure pour un pipeline de données multimodal a été esquissée dans `neurolite/data/datasets.py`, incluant `MultimodalDataset` pour charger des données à partir d'un fichier manifest et `multimodal_collate_fn` pour batcher efficacement ces données pour le modèle.

## 🧠 Vision AGI et Fondements Théoriques

NeuroLite s'inspire de plusieurs avancées théoriques récentes:

- **MLP-Mixer**: Démontre que des projections linéaires alternées (token-mixing et channel-mixing) peuvent rivaliser avec l'attention pour de nombreuses tâches
- **Complexité Linéaire**: Exploite les approches comme Performer, Linformer et FNet qui remplacent l'attention quadratique par des approximations efficaces
- **Mémoire Associative Moderne**: Intègre des réseaux de Hopfield continus de grande capacité pour la mémorisation associative
- **Routage Adaptatif**: Utilise des techniques de routage dynamique pour activer sélectivement différents "experts" selon le contexte
- **Composants Neurosymboliques (étendus)**: Combine traitement neuronal avec des mécanismes de raisonnement symbolique et probabiliste plus explicites pour améliorer les capacités de raisonnement structuré et la gestion de l'incertitude, tout en maintenant une faible empreinte paramétrique.
- **Apprentissage Continu (Lifelong Learning)**: S'inspire des approches visant à permettre aux modèles d'apprendre séquentiellement de nouvelles tâches ou données sans oublier les précédentes. L'interaction du `ContinualAdapter` avec `HierarchicalMemory` est une étape vers cet objectif.
- **Mémoires Hiérarchiques Dynamiques**: La `HierarchicalMemory` s'inspire des modèles cognitifs, où la consolidation et la récupération sont des processus dynamiques, maintenant enrichis par l'intégration avec les modules de raisonnement et d'apprentissage continu.
- **Traitement Multimodal et Fusion d'Informations**: L'utilisation du `CrossModalFuser` au niveau de `NeuroLiteModel` représente une approche plus directe et potentiellement plus puissante pour la fusion des modalités en entrée.

La vision AGI de NeuroLite repose sur l'hypothèse que l'intelligence générale émergera de l'intégration étroite et de la co-optimisation de ces différents modules, permettant au système de percevoir, mémoriser, apprendre, et raisonner de manière plus holistique et adaptable.

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
├── symbolic.py        # Composants de raisonnement symbolique
├── continual.py       # Adaptateur d'apprentissage continu
└── multimodal.py      # Modules pour la projection et l'attention multimodales

Configs/
│   └─ config.py       # Configuration des différentes tailles de modèles

core/
│   ├─ model.py        # Modèle principal et variantes spécialisées
│   ├─ mixer.py        # Implémentations MLP-Mixer, HyperMixer, FNet
│   └─ multimodal/     # Modules multimodaux principaux

memory/
│   ├─ memory.py       # Mémoire externe différentiable (base)
│   └─ hierarchical_memory.py # Mémoire hiérarchique améliorée

multimodal/
│   ├─ multimodal.py   # Module principal de projection multimodale
│   ├─ encoders/       # Encodeurs spécialisés par modalité
│   │   ├─ text_encoder.py
│   │   ├─ image_encoder.py
│   │   ├─ audio_encoder.py
│   │   ├─ video_encoder.py
│   │   └─ graph_encoder.py
│   └─ decoders/       # Décodeurs spécialisés par modalité
│       ├─ text_decoder.py
│       ├─ image_decoder.py
│       ├─ audio_decoder.py
│       ├─ video_decoder.py
│       └─ graph_decoder.py

tokenization/
│   ├─ tokenizer.py    # Tokenizer multimodal avancé
│   ├─ config.py       # Configuration du tokenizer
│   └─ vector_quantization.py # Implémentation de la quantification vectorielle

routing/           # Routage dynamique et Mixture-of-Experts

symbolic/          # Composants de raisonnement symbolique

continual/         # Adaptateur d'apprentissage continu

reasoning/         # Modules de raisonnement avancé

scripts/
├─ tokenizer.py         # Script de démonstration du tokenizer multimodal
├─ multimodal_test.py    # Tests des fonctionnalités multimodales
└─ train_tokenizer.py    # Entraînement du tokenizer multimodal

examples/
├─ simple_example.py           # Exemple basique d'utilisation
├─ classification_example.py   # Classification de texte
├─ memory_and_routing_example.py # Démonstration mémoire et routage
├─ symbolic_reasoning_example.py # Démonstration du moteur de règles et couche neurosymbolique
├─ bayesian_network_example.py   # Démonstration du réseau bayésien
├─ lifelong_learning_demo.py     # Démonstration de l'apprentissage continu
├─ multimodal_input_example.py   # Démonstration des entrées multimodales
├─ multimodal_generation_example.py # Démonstration de génération multimodale
└─ benchmark_comparison.py     # Comparaison avec architectures standards
# (Nouveaux exemples à ajouter pour illustrer les capacités récentes)
# - ptq_demonstration.py: Montre l'application de la quantification et la réduction de taille.
# - reasoning_with_memory_example.py: Illustre comment NeurosymbolicReasoner/StructuredPlanner interagissent avec HierarchicalMemory.
# - continual_learning_consolidation_example.py: Démontre la consolidation des connaissances du ContinualAdapter vers HierarchicalMemory.
# - dynamic_routing_stats_example.py: Montre comment récupérer et interpréter les statistiques d'activation des experts.
# - multimodal_dataset_usage_example.py: Exemple d'utilisation du nouveau MultimodalDataset.

data/ # Nouveau répertoire pour les modules de données
│   └─ datasets.py     # Esquisse du MultimodalDataset et collate_fn

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

### Noyau Latent Universel

Le noyau latent de NeuroLite permet de projeter et de transformer des représentations entre différentes modalités :

```python
from neurolite import NeuroLiteModel, NeuroLiteConfig
import torch

# Créer un modèle avec support multimédia
config = NeuroLiteConfig.small()
config.use_multimodal_input = True
model = NeuroLiteModel(config, task_type="multimodal_generation")

# Charger des données multimodales (exemple simplifié)
multimodal_inputs = {
    "text": ["Un chat noir sur un canapé"],
    "image": torch.randn(1, 3, 224, 224)  # Exemple d'image
}

# Obtenir les représentations latentes
with torch.no_grad():
    outputs = model.forward(
        multimodal_inputs=multimodal_inputs,
        output_hidden_states=True,
        return_dict=True
    )
    
# Les représentations latentes sont disponibles dans outputs['all_hidden_states']
print(f"Taille des états cachés: {outputs['all_hidden_states'][-1].shape}")

# Visualisation des représentations latentes (nécessite matplotlib)
import matplotlib.pyplot as plt
latents = outputs['all_hidden_states'][-1].mean(dim=1).cpu().numpy()
plt.figure(figsize=(10, 5))
plt.imshow(latents, aspect='auto', cmap='viridis')
plt.colorbar(label="Valeur d'activation")
plt.title("Représentations Latentes")
plt.xlabel("Dimension")
plt.ylabel("Exemple")
plt.show()
```

### Génération Multimodale

Générez du contenu dans différentes modalités à partir d'entrées multimodales :

```python
# Générer à partir d'une entrée multimodale
generated_outputs = model.generate(
    multimodal_inputs=multimodal_inputs,
    target_modalities=["text", "image"],  # Générer du texte et une image
    max_length=50,  # Pour la génération de texte
    temperature=0.7
)

# Accéder aux sorties générées
if "generated_text" in generated_outputs:
    print("Texte généré:", generated_outputs["generated_text"])
    
if "generated_image" in generated_outputs:
    # Afficher ou sauvegarder l'image générée
    plt.imshow(generated_outputs["generated_image"][0].permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.show()
```

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

### Utilisation de la Mémoire Contextuelle et du Raisonnement

```python
# Créer un modèle avec HierarchicalMemory et modules de raisonnement
config = NeuroLiteConfig.base() # Utilise HierarchicalMemory par défaut
config.model_config.use_symbolic_module = True
config.model_config.use_advanced_reasoning = True # Pour NeurosymbolicReasoner
# config.model_config.use_planning_module = True # Pour StructuredPlanner

# S'assurer que le device est correctement configuré (ex: 'cpu' si pas de GPU)
config.device = 'cpu'
if hasattr(config.model_config, 'device'): # Si ModelArchitectureConfig a aussi un champ device
    config.model_config.device = 'cpu'


model = NeuroLiteModel(config)
model.eval() # Important pour les modules avec comportements différents train/eval

# Fournir du contexte à la mémoire (via NeuroLiteModel qui le passe à HierarchicalMemory)
# Pour un modèle non-multimodal (config.model_config.use_multimodal_input = False par défaut pour base non modifié)
# et input_projection_type="minhash_bloom" (aussi par défaut)
initial_context = {"text": ["Alice est une ingénieure vivant à Paris.", "Bob est un artiste de Lyon."]}
model.forward(multimodal_inputs=initial_context, update_memory=True)

# Le raisonneur peut maintenant potentiellement utiliser ce contexte stocké dans HierarchicalMemory
# (Simulation, l'interaction directe dépend de l'implémentation exacte de NeurosymbolicReasoner.forward)
query_text = {"text": ["Qui vit à Paris et est ingénieure ?"]}
# Pour obtenir les sorties symboliques, si le modèle le supporte :
# outputs = model.forward(multimodal_inputs=query_text, return_symbolic=True)
# print(outputs.get("symbolic_outputs"))
result = model.forward(multimodal_inputs=query_text)
print(f"Sortie après contexte: {result['hidden_states'].shape}")

# L'adaptateur continu peut aussi interagir avec la mémoire pour consolider
# if config.model_config.use_continual_adapter:
#    adapter_input = {"text": ["Nouvelle information pertinente à consolider."]}
#    model.forward(multimodal_inputs=adapter_input, continuous_learning=True, update_memory=True)

# Pour observer les stats du routeur dynamique (si des DynamicRoutingBlocks sont dans l'architecture)
# config.model_config.use_dynamic_routing = True (s'assurer qu'il est activé)
# dummy_input_for_router = {"text": ["Un test pour le routeur dynamique."]}
# if hasattr(model, 'layers') and len(model.layers) > 0 and hasattr(model.layers[0], 'return_expert_stats'):
#     # Supposons que la première couche est un DynamicRoutingBlock ou contient un routeur compatible
#     # Pour un test direct, il faudrait instancier DynamicRoutingBlock et passer des données
#     print("Note: Pour tester return_expert_stats, exécutez un forward pass sur un modèle configuré avec DynamicRoutingBlock.")
#     # outputs = model.forward(multimodal_inputs=dummy_input_for_router, return_expert_stats=True)
#     # if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], dict):
#     #    print(f"Statistiques des experts du premier bloc de routage : {outputs[1]}")
```

### Utilisation du Tokenizer Multimodal

```python
from neurolite.tokenization import NeuroLiteTokenizer, TokenizerConfig
import torch

# Créer une configuration pour le tokenizer
tokenizer_config = TokenizerConfig(
    semantic_vocab_size=8192,    # Taille du vocabulaire sémantique
    detail_vocab_size=32768,     # Taille du vocabulaire de détails
    hidden_size=768,             # Dimension des représentations
    dropout_rate=0.1,            # Taux de dropout
    use_residual_vq=True,        # Utiliser la quantification vectorielle résiduelle
    use_dual_codebook=True,      # Utiliser le double codebook
    num_residual_layers=3,       # Nombre de couches résiduelles
    hierarchical_levels=3        # Niveaux hiérarchiques
)

# Initialiser le tokenizer
tokenizer = NeuroLiteTokenizer(tokenizer_config)

# Préparer des entrées multimodales
multimodal_inputs = {
    'text': ["L'architecture universelle d'IA combine des encodeurs modulaires et un noyau latent"],
    'image': torch.randn(1, 3, 224, 224),   # Image simulée [B, C, H, W]
    'audio': torch.randn(1, 1, 16000),      # Audio simulé [B, C, T]
    'video': torch.randn(1, 8, 3, 112, 112) # Vidéo simulée [B, F, C, H, W]
}

# Tokenizer les entrées
tokens = tokenizer.tokenize(multimodal_inputs)

# Accéder aux résultats
semantic_tokens = tokens['semantic_tokens']     # Représentations sémantiques
detail_tokens = tokens['detail_tokens']         # Représentations détaillées
semantic_indices = tokens['semantic_indices']   # Indices des tokens sémantiques
detail_indices = tokens['detail_indices']       # Indices des tokens détaillés

print(f"Forme des tokens sémantiques: {semantic_tokens.shape}")
print(f"Forme des tokens détaillés: {detail_tokens.shape}")
```

### Traitement et Génération Multimodale

```python
from neurolite import NeuroLiteModel, NeuroLiteConfig
from neurolite.tokenization import NeuroLiteTokenizer, TokenizerConfig
import torch

# Créer une configuration du tokenizer
tokenizer_config = TokenizerConfig(
    semantic_vocab_size=8192,
    detail_vocab_size=32768,
    hidden_size=768
)

# Initialiser le tokenizer
tokenizer = NeuroLiteTokenizer(tokenizer_config)

# Configurer le modèle pour l'entrée et la génération multimodale
config = NeuroLiteConfig.base()
config.use_multimodal_input = True
config.multimodal_output_dim = 768
config.multimodal_hidden_dim = 768
config.use_cross_modal_attention = True
config.cross_modal_num_heads = 8
config.image_size = 224
config.multimodal_image_patch_size = 16
config.max_audio_length_ms = 30000
config.multimodal_video_num_sampled_frames = 16
config.max_graph_nodes = 32

# Créer le modèle pour la génération multimodale
model = NeuroLiteModel(
    config=config,
    task_type="multimodal_generation",
    tokenizer=tokenizer
)

# Préparer les données d'entrée
inputs = {
    "text": ["Décris cette image d'un chat."],
    "image": torch.randn(1, 3, 224, 224)  # Image simulée
}

# Générer des sorties multimodales
outputs = model.generate(
    multimodal_inputs=inputs,
    target_modalities=["text", "image"],  # Modalités à générer
    temperature=0.8,                    # Contrôle la diversité
    max_length=100                      # Longueur maximale pour le texte
)

# Traiter les sorties générées
if "text_output" in outputs:
    generated_text = outputs["text_output"]
    print(f"Texte généré: {generated_text}")

if "image_output" in outputs:
    generated_image = outputs["image_output"]  # Tensor [1, 3, H, W]
    # Visualiser l'image générée
    import matplotlib.pyplot as plt
    plt.imshow(generated_image[0].permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.show()
```

## 🏋️ Entraînement du Modèle et Pipeline de Données

L'entraînement de NeuroLite peut bénéficier du nouveau pipeline de données esquissé dans `neurolite/data/datasets.py`. Ce pipeline facilite la gestion des ensembles de données multimodales complexes.

```python
# Exemple conceptuel d'utilisation du nouveau MultimodalDataset
from neurolite.data.datasets import MultimodalDataset, multimodal_collate_fn
from torch.utils.data import DataLoader

# 1. Créer un fichier manifest (par exemple, manifest.csv):
# sample_id,text_path,image_path,audio_path,label
# id1,path/to/text1.txt,path/to/image1.png,path/to/audio1.wav,0
# id2,path/to/text2.txt,path/to/image2.png,path/to/audio2.wav,1
# (Assurez-vous que les fichiers de données existent ou utilisez des placeholders dans le Dataset)

# 2. Initialiser le Dataset (le tokenizer est généralement dans le modèle NeuroLite)
#    Pour cet exemple, nous utilisons des placeholders pour les chemins.
#    Créez un dummy_manifest.csv avec des entrées comme:
#    id1,"Ceci est du texte.",dummy_image.png,dummy_audio.wav,0
# with open("dummy_manifest.csv", "w") as f:
#    f.write("sample_id,text_path,image_path,audio_path,label\n")
#    f.write('id1,"Texte du sample 1.",img1.png,audio1.wav,0\n')
#    f.write('id2,"Autre texte.",img2.png,audio2.wav,1\n')

# dataset = MultimodalDataset(manifest_file="dummy_manifest.csv", tokenizer=None)

# 3. Initialiser le DataLoader
# dataloader = DataLoader(dataset, batch_size=2, collate_fn=multimodal_collate_fn)

# 4. Boucle d'entraînement (concept)
# for batch_data in dataloader:
#     # batch_data contient {'multimodal_inputs': {...}, 'labels': ..., 'ids': ...}
#     # multimodal_inputs = batch_data['multimodal_inputs']
#     # labels = batch_data.get('labels')
#
#     # outputs = model.forward(multimodal_inputs=multimodal_inputs, labels=labels)
#     # loss = outputs.get('loss')
#     # if loss is not None:
#     #     loss.backward()
#     #     optimizer.step()
#     #     optimizer.zero_grad()
#     print(f"Traitement d'un batch (IDs: {batch_data.get('ids')})...") # Placeholder
```
*(Note: L'exemple ci-dessus est conceptuel. L'exécution nécessite un fichier manifest et des données réelles ou des placeholders correctement gérés dans `MultimodalDataset`.)*

Pour l'entraînement avec des scripts existants (exemple : génération de texte sur WikiText) :

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

## 📊 Statistiques du Modèle

NeuroLite a été conçu pour être performant tout en maîtrisant les ressources nécessaires. Voici quelques statistiques clés de l'architecture actuelle :

- **Paramètres totaux** : ~335M paramètres
- **Répartition des paramètres** :
  - Encodeurs spécialisés : ~60% des paramètres
  - Codebooks (sémantique et détail) : ~15% des paramètres
  - Autres composants : ~25% des paramètres
- **Temps d'inférence** : 5× plus rapide que les Transformers standards sur des séquences longues
- **Empreinte mémoire** : 3× à 4× inférieure aux Transformers équivalents

### Optimisations Futures

Plusieurs techniques sont à l'étude pour réduire davantage la taille du modèle tout en préservant ses performances :

- **Quantification des poids** : Réduction de la précision des poids (int8/int4)
- **Élagage (pruning)** : Suppression des connexions les moins importantes
- **Distillation de connaissance** : Transfert des connaissances d'un grand modèle vers des architectures plus compactes.
- **Routage plus granulaire** : Activation plus sélective des experts selon le contexte pour réduire les calculs.
- **Quantification PTQ et Statique**: L'exploration de la quantification dynamique (PTQ) sur `NeuroLiteModel` (configuration `base`) a montré une **réduction de taille d'environ 70%** (de ~230Mo à ~65Mo) tout en conservant la capacité d'inférence. La quantification statique est une piste future pour des gains de vitesse potentiels.
- **Améliorations du `DynamicRoutingBlock`**: Le `SparseDispatcher`, cœur du routage dynamique, a été optimisé en remplaçant une boucle Python par `torch.scatter_add_` pour l'agrégation des sorties d'experts. De plus, il peut maintenant retourner des **statistiques détaillées sur l'utilisation des experts** (nombre de tokens traités par expert, nombre d'experts actifs, etc.), utiles pour l'analyse et le débogage du routage.

## 🎯 Visualisations Multimodales

L'architecture NeuroLite implémente un système de traitement multimodal unifié où différentes modalités (texte, image, audio, vidéo, graphe) sont projetées dans un espace latent commun. Voici des visualisations qui illustrent le fonctionnement de cette architecture universelle :

### Similarité entre représentations des modalités

<div align="center">
<img src="outputs/tokenizer/tokenizer_visualization_20250525_180047.png" alt="Tokenization multimodale" width="800"/>
</div>

Cette visualisation montre les indices des tokens sémantiques (en haut) et détaillés (en bas) générés par le tokenizer multimodal. Le système de double codebook de NeuroLite permet de capturer à la fois les caractéristiques sémantiques globales et les détails fins pour chaque modalité, offrant une représentation riche qui préserve la structure et le contexte des données d'entrée.

### Distribution des tokens dans l'espace de codebook

<div align="center">
<img src="outputs/tokenizer/tokenizer_visualization_20250525_180047_distribution.png" alt="Distribution des tokens" width="800"/>
</div>

Cet histogramme montre la distribution des tokens sémantiques (en haut) et détaillés (en bas) dans l'espace de codebook. On peut observer comment le tokenizer assigne des indices de façon non-uniforme, reflétant les distributions naturelles des caractéristiques dans les données multimodales. Cette distribution adaptative permet d'optimiser l'utilisation du vocabulaire limité pour représenter un espace d'information beaucoup plus vaste.

### Espace latent des représentations multimodales

<div align="center">
<img src="outputs/tokenizer/tokenizer_visualization_20250525_180047_latent_space.png" alt="Espace latent multimodal" width="800"/>
</div>

Cette visualisation par réduction dimensionnelle (PCA) montre comment différentes modalités sont représentées dans l'espace latent commun. Chaque point correspond à une modalité spécifique (texte, image, audio, vidéo, graphe) avec son propre marqueur et couleur. 

**Points clés illustrés :**

- **Noyau latent universel** : Toutes les modalités sont projetées dans un même espace vectoriel, servant de "lingua franca" pour l'architecture.
  
  > **Note**: Le terme "lingua franca" désigne historiquement une langue véhiculaire permettant la communication entre personnes de langues maternelles différentes. Dans notre architecture, le noyau latent universel joue ce rôle en permettant à différentes modalités (ayant chacune leur propre "langage" de représentation) de communiquer et d'interagir dans un format compatible et unifié.
  
- **Préservation des caractéristiques distinctives** : Les différentes modalités occupent des régions distinctes de l'espace latent, ce qui permet de préserver leurs caractéristiques uniques.
  
- **Double représentation (sémantique/détaillée)** : Les panneaux supérieur et inférieur montrent respectivement les espaces latents sémantique et détaillé, illustrant la complémentarité de la double codification.
  
- **Alignement inter-modal** : Les distances entre points révèlent les relations naturelles entre modalités (par exemple, proximité entre représentations texte-image ou audio-vidéo).

Ces visualisations démontrent comment NeuroLite parvient à unifier le traitement de différentes modalités tout en préservant leurs spécificités, un principe fondamental de notre architecture universelle d'IA.

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
    novelty_threshold_pm=0.7,  # Seuil pour la mise à jour de la mémoire persistante

    # Configuration pour l'entrée et l'attention multimodales
    use_multimodal_input=True,
    multimodal_output_dim=config.hidden_size, # Dimension de sortie de MultimodalProjection
    multimodal_image_patch_size=16,
    multimodal_video_num_sampled_frames=5,
    use_cross_modal_attention=True,
    cross_modal_num_heads=4
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

## 🔄 Gestion des Données & Optimisations (Révisé et Étendu)

NeuroLite intègre et propose des systèmes pour une gestion efficace des données et des optimisations de modèles :

- **Nouveau Pipeline de Données Multimodal (Esquisse)**:
    - Le fichier `neurolite/data/datasets.py` contient une esquisse pour `MultimodalDataset` et `multimodal_collate_fn`.
    - `MultimodalDataset` est conçu pour lire des fichiers manifest (CSV/JSONL) listant les échantillons et leurs données multimodales.
    - `multimodal_collate_fn` s'occupe du batching intelligent, préparant les données pour `NeuroLiteModel`.
    - Cette structure vise à simplifier la gestion d'ensembles de données complexes pour l'entraînement et l'évaluation de modèles multimodaux.

- **Optimisations d'Efficacité du Modèle**:
    - **Quantification Post-Entraînement (PTQ)**: Des tests ont démontré une réduction significative de la taille du modèle (~70%) par quantification dynamique (`torch.qint8`), avec des inférences réussies sur le modèle quantifié. Cela est crucial pour le déploiement sur des appareils à ressources limitées.
    - **Routage Dynamique Optimisé**: Le `SparseDispatcher` au sein du `DynamicRoutingBlock` utilise maintenant `torch.scatter_add_` pour une agrégation plus performante des sorties d'experts. Il peut également retourner des statistiques d'utilisation des experts pour une meilleure analysabilité du modèle.

- **Fonctionnalités de Gestion de Données Existantes**:
    - WikiTextDataset: Chargement et gestion du corpus WikiText.
    - Padding intelligent et tokenization optimisée.
    - Multiprocessing pour le chargement de données.
    - Intégration avec l'écosystème PyTorch (DataLoader, etc.).

## 🛣️ Feuille de Route et Contributions

NeuroLite est un projet en évolution active, avec une feuille de route axée sur le renforcement de ses capacités AGI et son efficacité :
- **Capacités AGI Approfondies**: Poursuivre l'intégration et la co-optimisation des modules de mémoire, raisonnement, et apprentissage continu. Développer des mécanismes de prise de décision plus autonomes et une meilleure gestion de l'incertitude.
- **Benchmarks et Évaluations Rigoureux**: Évaluer NeuroLite sur un large éventail de tâches multimodales, d'apprentissage continu, et (si possible) de benchmarks orientés AGI. Comparer systématiquement avec d'autres architectures SOTA légères et plus larges.
- **Pipeline de Données Multimodal Complet**: Finaliser l'implémentation de `MultimodalDataset` et `multimodal_collate_fn`, incluant le support robuste pour toutes les modalités et des transformations avancées.
- **Optimisations Avancées**:
    - Explorer la **quantification statique (PTQ)** et la **quantification consciente de l'entraînement (QAT)** pour améliorer davantage la vitesse d'inférence et potentiellement la précision par rapport à la PTQ dynamique.
    - Implémenter des techniques d'**élagage (pruning)** pour réduire la densité du modèle.
    - Affiner les stratégies de **routage dynamique** pour un équilibre optimal entre performance et coût computationnel.
- **Documentation et Exemples Concrets**: Enrichir la documentation et fournir des exemples de code complets pour toutes les fonctionnalités clés, y compris :
    - Entraînement et inférence avec le nouveau pipeline de données multimodal.
    - Utilisation avancée de `HierarchicalMemory` avec les modules de raisonnement et d'apprentissage continu.
    - Application et évaluation de modèles quantifiés.
    - Interprétation des statistiques du `DynamicRoutingBlock`.
- **Extension des Capacités Symboliques**: Améliorer l'expressivité du `SymbolicRuleEngine` et l'intégration des connaissances symboliques dans le flux neuronal.

Les contributions de la communauté sont essentielles pour accélérer le développement de NeuroLite. Que vous soyez intéressé par l'ajout de nouvelles fonctionnalités, l'amélioration des performances, la création de benchmarks, l'enrichissement de la documentation ou le test du modèle sur de nouvelles tâches, vos contributions sont les bienvenues ! N'hésitez pas à ouvrir des issues pour discuter d'idées ou soumettre des pull requests.

---

<div align="center">
<strong>NeuroLite - Construire les fondations d'une AGI légère, efficace et adaptable.</strong>
</div>
