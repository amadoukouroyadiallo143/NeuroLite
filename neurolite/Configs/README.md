# Configuration de NeuroLite

Ce dossier contient les configurations pour les modèles et composants de NeuroLite.

## Structure des configurations

### Fichiers principaux

- `config.py` : Contient toutes les classes de configuration pour NeuroLite
- `__init__.py` : Fichier d'initialisation du module

## Utilisation des configurations

### Création d'une configuration personnalisée

```python
from neurolite.Configs.config import NeuroLiteConfig, ModelArchitectureConfig, TokenizerConfig

# Créer une configuration personnalisée
config = NeuroLiteConfig(
    model_config=ModelArchitectureConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
    ),
    tokenizer_config=TokenizerConfig(
        semantic_vocab_size=16384,
        hidden_size=768
    )
)
```

### Configurations prédéfinies

NeuroLite fournit plusieurs configurations prédéfinies :

```python
from neurolite.Configs.config import NeuroLiteConfig

# Modèle léger
config = NeuroLiteConfig.tiny()  # ~1-2Mo

# Modèle petit
config = NeuroLiteConfig.small()  # ~5-10Mo

# Modèle de base (par défaut)
config = NeuroLiteConfig.base()  # ~20-30Mo

# Grand modèle
config = NeuroLiteConfig.large()  # ~100-200Mo

# Modèle avec raisonnement symbolique
config = NeuroLiteConfig.base_symbolic()
```

### Sauvegarde et chargement

```python
# Sauvegarder la configuration
config.save_pretrained("mon_modele")

# Charger une configuration
config = NeuroLiteConfig.from_pretrained("mon_modele")
```

## Structure des configurations

### Configuration principale (NeuroLiteConfig)

Contient les sous-configurations pour tous les composants du modèle :

- `model_config` : Architecture du modèle
- `training_config` : Paramètres d'entraînement
- `logging_config` : Configuration de la journalisation
- `memory_config` : Configuration de la mémoire
- `tokenizer_config` : Configuration du tokenizer

### Configuration du tokenizer (TokenizerConfig)

Gère la configuration pour le tokenizer multimodal, y compris :

- Paramètres de vocabulaire
- Configurations spécifiques aux encodeurs (texte, image, audio, vidéo, graphe)
- Paramètres de quantification vectorielle
- Paramètres d'entraînement spécifiques

## Bonnes pratiques

1. **Héritage** : Toutes les classes de configuration héritent de `BaseConfig`
2. **Typage** : Utilisez les types Python pour la validation
3. **Documentation** : Documentez tous les paramètres avec des docstrings
4. **Rétrocompatibilité** : Maintenez la compatibilité avec les configurations existantes

## Exemple complet

```python
from neurolite.Configs.config import (
    NeuroLiteConfig,
    ModelArchitectureConfig,
    TokenizerConfig,
    TextEncoderConfig,
    ImageEncoderConfig
)

# Créer une configuration personnalisée
config = NeuroLiteConfig(
    model_config=ModelArchitectureConfig(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
    ),
    tokenizer_config=TokenizerConfig(
        semantic_vocab_size=16384,
        detail_vocab_size=65536,
        text_encoder_config=TextEncoderConfig(
            vocab_size=100000,
            hidden_size=1024
        ),
        vision_encoder_config=ImageEncoderConfig(
            image_size=256,
            patch_size=16
        )
    ),
    training_config=TrainingConfig(
        learning_rate=5e-5,
        batch_size=32,
        num_train_epochs=10
    )
)
```

## Dépannage

### Problèmes courants

1. **Erreurs de type** : Vérifiez que tous les paramètres ont des types corrects
2. **Valeurs manquantes** : Les valeurs par défaut sont définies dans les classes de configuration
3. **Compatibilité** : Assurez-vous que les versions des configurations correspondent à la version de NeuroLite

## Contribution

Les contributions pour améliorer le système de configuration sont les bienvenues. Assurez-vous de :

1. Mettre à jour la documentation
2. Ajouter des tests unitaires
3. Maintenir la rétrocompatibilité
4. Suivre les conventions de code existantes
