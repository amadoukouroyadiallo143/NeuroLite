# 🚀 NeuroLite AGI Training v2.0

Système d'entraînement complet pour NeuroLite AGI avec support modulaire, multi-GPU et métriques AGI spécialisées.

## 🏃‍♂️ Démarrage Rapide

### Test Ultra-Rapide (2 minutes)
```bash
# Test configuration (dry run)
python -m neurolite.train_neurolite --dry-run --config tiny

# Test entraînement (1 batch)
python -m neurolite.train_neurolite --fast-dev-run --config tiny --batch-size 1
```

### Entraînement Complet Local
```bash
# Modèle tiny (recommandé pour budget 0€)
python -m neurolite.train_neurolite \
    --config tiny \
    --epochs 3 \
    --batch-size 2 \
    --dataset-type multimodal \
    --experiment-name "neurolite_tiny_test"
```

### Avec Wandb (optionnel)
```bash
# Avec monitoring Wandb
python -m neurolite.train_neurolite \
    --config tiny \
    --use-wandb \
    --experiment-name "neurolite_wandb_test"
```

## 📊 Datasets Disponibles

| Dataset | Taille | Description | Utilisation |
|---------|--------|-------------|-------------|
| `multimodal` | 800 échantillons | Texte + Image + Audio | Entraînement général |
| `text` | 500 séquences | Génération de texte | Phase génération |
| `consciousness` | 300 échantillons | Niveaux de conscience | Module conscience |
| `memory` | 400 échantillons | Rappel mémoire | Module mémoire |

## 🎯 Phases d'Entraînement

Le système utilise 5 phases d'entraînement séquentielles :

### 1. Foundation (3 époques)
```bash
# Entraîne SSM Core + Multimodal Fusion
python -m neurolite.train_neurolite --phase foundation
```

### 2. Cognition (2 époques) 
```bash
# Entraîne Cognitive Core
python -m neurolite.train_neurolite --phase cognition
```

### 3. Higher-Order (2 époques)
```bash
# Entraîne Consciousness + Memory
python -m neurolite.train_neurolite --phase higher_order
```

### 4. Integration (1 époque)
```bash
# Entraîne Reasoning + Interface
python -m neurolite.train_neurolite --phase integration
```

### 5. End-to-End (3 époques)
```bash
# Entraînement global de tous les modules
python -m neurolite.train_neurolite --phase end_to_end
```

## ⚙️ Configurations

### Tiny (Recommandé - Budget 0€)
- **Paramètres** : ~12M
- **RAM** : 2GB
- **VRAM** : 4GB
- **Temps** : 30min sur CPU, 5min sur GPU

### Development
- **Paramètres** : ~25M  
- **RAM** : 4GB
- **VRAM** : 8GB
- **Temps** : 1h sur CPU, 15min sur GPU

## 🔧 Options Avancées

### Optimisation Mémoire
```bash
# Ultra-compact pour machines limitées
python -m neurolite.train_neurolite \
    --config tiny \
    --batch-size 1 \
    --accumulate-grad-batches 8
```

### GPU Multiple (si disponible)
```bash
# Multi-GPU
python -m neurolite.train_neurolite \
    --gpus 2 \
    --batch-size 4
```

### Debug & Development
```bash
# Profiling activé
python -m neurolite.train_neurolite \
    --config dev \
    --log-every-n-steps 1 \
    --fast-dev-run
```

## 📈 Métriques Suivies

### Métriques Générales
- **Loss** : Perte totale composite
- **Accuracy** : Précision approximative 
- **Perplexity** : Perplexité du modèle

### Métriques AGI Spécialisées
- **Consciousness Coherence** : Cohérence de conscience (0-1)
- **Memory Precision/Recall** : Précision mémoire
- **Reasoning Validity** : Validité du raisonnement
- **Processing Speed** : Vitesse de traitement

## 💾 Checkpoints & Sauvegarde

### Checkpoints Automatiques
- **Meilleur modèle** : Basé sur validation loss
- **Dernière époque** : Sauvegarde finale
- **Top-3** : 3 meilleurs checkpoints

### Localisation
```
./experiments/results/
├── {experiment_name}/
│   ├── checkpoints/
│   │   ├── best.ckpt
│   │   ├── last.ckpt
│   │   └── epoch_*.ckpt
│   └── logs/
│       └── tensorboard/
```

### Chargement Checkpoint
```python
from neurolite.training.trainer import NeuroLiteTrainer

# Charger checkpoint
model = NeuroLiteTrainer.load_from_checkpoint("path/to/checkpoint.ckpt")
```

## 🐛 Troubleshooting

### Erreur Mémoire GPU
```bash
# Réduire batch size
python -m neurolite.train_neurolite --batch-size 1

# Activer CPU offload
python -m neurolite.train_neurolite --config tiny
```

### Slow Training
```bash
# Test avec un seul batch
python -m neurolite.train_neurolite --fast-dev-run

# Vérifier GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Dataset Error
```bash
# Test dataset seul
python -c "from neurolite.datasets import create_sample_dataloaders; create_sample_dataloaders(batch_size=1)"
```

## 📊 Monitoring

### TensorBoard (Local)
```bash
# Lancer TensorBoard
tensorboard --logdir ./experiments/results

# Naviguer vers http://localhost:6006
```

### Wandb (Cloud - Optionnel)
```bash
# Setup Wandb
pip install wandb
wandb login

# Entraînement avec Wandb
python -m neurolite.train_neurolite --use-wandb
```

## 🎯 Cas d'Usage Typiques

### Test Rapide Development
```bash
python -m neurolite.train_neurolite \
    --fast-dev-run \
    --config tiny \
    --batch-size 1
```

### Entraînement Production Locale
```bash
python -m neurolite.train_neurolite \
    --config tiny \
    --epochs 5 \
    --batch-size 2 \
    --use-wandb \
    --experiment-name "prod_$(date +%Y%m%d)"
```

### Entraînement Colab
```bash
# Dans Colab
!pip install pytorch-lightning wandb
!python -m neurolite.train_neurolite \
    --config tiny \
    --batch-size 1 \
    --epochs 3 \
    --gpus 1
```

## 🔄 Pipeline Complet

```bash
# 1. Test setup
python -m neurolite.train_neurolite --dry-run

# 2. Test rapide
python -m neurolite.train_neurolite --fast-dev-run

# 3. Entraînement complet
python -m neurolite.train_neurolite --epochs 5

# 4. Évaluation
python -c "from neurolite.training.metrics import AGIMetrics; print('Métriques calculées')"
```

## 📝 Logs & Outputs

### Structure des Logs
```
./experiments/
├── results/
│   └── {experiment}/
│       ├── checkpoints/
│       └── version_*/
│           └── metrics.csv
└── logs/
    └── training.log
```

### Métriques Exportées
- **CSV** : `metrics.csv` avec toutes les métriques par step
- **JSON** : Résumé final de l'expérience
- **TensorBoard** : Graphiques interactifs

---

## 🎉 Exemple Complet

```bash
# Entraînement NeuroLite AGI - Configuration optimale budget 0€
python -m neurolite.train_neurolite \
    --config tiny \
    --dataset-type multimodal \
    --epochs 3 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --experiment-name "neurolite_budget_zero" \
    --log-every-n-steps 5
```

**Temps estimé** : 15-30 minutes sur CPU, 3-5 minutes sur GPU  
**Ressources** : 2GB RAM, 1GB VRAM  
**Résultat** : Modèle AGI fonctionnel de 12M paramètres