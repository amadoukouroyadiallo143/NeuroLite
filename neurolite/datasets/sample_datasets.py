"""
NeuroLite Sample Datasets v2.0
Datasets échantillons légers pour tests d'entraînement sur ordinateur personnel.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import random
import logging
from pathlib import Path
import json

from ..core.agi_model import AGIMode

logger = logging.getLogger(__name__)

class SampleMultimodalDataset(Dataset):
    """
    Dataset échantillon multimodal pour tests rapides.
    Génère des données synthétiques légères pour toutes les modalités.
    """
    
    def __init__(
        self,
        size: int = 1000,
        sequence_length: int = 32,
        hidden_size: int = 256,
        num_modalities: int = 3,
        seed: int = 42
    ):
        super().__init__()
        
        self.size = size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        
        # Générer données synthétiques
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        self.data = self._generate_synthetic_data()
        
        logger.info(f"📊 SampleMultimodalDataset créé: {size} échantillons")
    
    def _generate_synthetic_data(self) -> List[Dict[str, Any]]:
        """Génère les données synthétiques."""
        
        data = []
        tasks = [
            "Classification multimodale",
            "Génération de texte", 
            "Reconnaissance d'images",
            "Compréhension audio",
            "Raisonnement logique",
            "Synthèse d'informations"
        ]
        
        for i in range(self.size):
            # Modalités d'entrée
            inputs = {}
            
            # Texte (toujours présent)
            text_seq = torch.randn(self.sequence_length, self.hidden_size)
            inputs["text"] = text_seq
            
            # Image (50% du temps)
            if random.random() > 0.5:
                image_tensor = torch.randn(3, 64, 64)  # Petite image RGB
                inputs["image"] = image_tensor
            
            # Audio (30% du temps)  
            if random.random() > 0.7:
                audio_tensor = torch.randn(1, 1024)  # Court échantillon audio
                inputs["audio"] = audio_tensor
            
            # Target principal (basé sur texte avec bruit)
            primary_target = text_seq + torch.randn_like(text_seq) * 0.1
            
            # Métadonnées AGI
            consciousness_target = random.uniform(0.3, 0.8)
            
            # Chaîne de raisonnement synthétique
            reasoning_steps = random.randint(2, 6)
            reasoning_target = [f"Étape {j+1}: Analyse {j}" for j in range(reasoning_steps)]
            
            # Mode AGI aléatoire
            mode = random.choice(list(AGIMode))
            
            sample = {
                "task": random.choice(tasks),
                "inputs": inputs,
                "targets": {
                    "primary_target": primary_target,
                    "consciousness_target": consciousness_target,
                    "reasoning_targets": reasoning_target,
                    "memory_targets": {
                        "expected_count": random.randint(1, 5),
                        "relevance_threshold": 0.7
                    }
                },
                "mode": mode,
                "sample_id": i
            }
            
            data.append(sample)
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


class TinyTextDataset(Dataset):
    """Dataset texte ultra-léger pour tests rapides."""
    
    def __init__(
        self, 
        size: int = 500,
        vocab_size: int = 1000,
        sequence_length: int = 16
    ):
        super().__init__()
        
        self.size = size
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        
        # Phrases d'exemple simples
        self.base_sentences = [
            "Le chat mange la souris",
            "NeuroLite est une AGI révolutionnaire", 
            "L'intelligence artificielle évolue rapidement",
            "Les réseaux de neurones apprennent par expérience",
            "La conscience artificielle émerge de la complexité",
            "Les modèles transformers utilisent l'attention",
            "L'apprentissage automatique nécessite des données",
            "Les algorithmes optimisent leurs paramètres"
        ]
        
        self.data = self._generate_text_data()
        
        logger.info(f"📝 TinyTextDataset créé: {size} séquences texte")
    
    def _generate_text_data(self) -> List[Dict[str, Any]]:
        """Génère des données texte synthétiques."""
        
        data = []
        
        for i in range(self.size):
            # Sélection sentence de base
            base_sentence = random.choice(self.base_sentences)
            words = base_sentence.split()
            
            # Tokenisation simple (word-level)
            input_ids = []
            for word in words[:self.sequence_length]:
                token_id = hash(word) % self.vocab_size
                input_ids.append(token_id)
            
            # Padding si nécessaire
            while len(input_ids) < self.sequence_length:
                input_ids.append(0)  # PAD token
            
            input_tensor = torch.tensor(input_ids[:self.sequence_length])
            
            # Target = input décalé (language modeling)
            target_tensor = torch.roll(input_tensor, -1)
            target_tensor[-1] = 0  # EOS
            
            sample = {
                "task": "Génération de texte",
                "inputs": {"text": input_tensor},
                "targets": {"primary_target": target_tensor},
                "mode": AGIMode.CREATIVE,
                "original_text": base_sentence
            }
            
            data.append(sample)
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


class ConsciousnessDataset(Dataset):
    """Dataset pour entraîner le module de conscience."""
    
    def __init__(self, size: int = 300):
        super().__init__()
        
        self.size = size
        self.data = self._generate_consciousness_data()
        
        logger.info(f"🧠 ConsciousnessDataset créé: {size} échantillons conscience")
    
    def _generate_consciousness_data(self) -> List[Dict[str, Any]]:
        """Génère des données pour entraîner la conscience."""
        
        data = []
        
        # Scénarios de conscience
        scenarios = [
            {"context": "Réflexion simple", "target_level": 0.3},
            {"context": "Analyse complexe", "target_level": 0.6}, 
            {"context": "Créativité", "target_level": 0.8},
            {"context": "Méta-cognition", "target_level": 0.9},
            {"context": "Traitement automatique", "target_level": 0.1},
        ]
        
        for i in range(self.size):
            scenario = random.choice(scenarios)
            
            # Input représentant le contexte
            context_complexity = scenario["target_level"]
            noise_level = (1.0 - context_complexity) * 0.5
            
            input_tensor = torch.randn(32, 256) * context_complexity + torch.randn(32, 256) * noise_level
            
            # Target niveau de conscience
            target_consciousness = scenario["target_level"] + random.uniform(-0.1, 0.1)
            target_consciousness = max(0.0, min(1.0, target_consciousness))
            
            sample = {
                "task": f"Conscience: {scenario['context']}",
                "inputs": {"text": input_tensor},
                "targets": {
                    "consciousness_target": target_consciousness,
                    "primary_target": input_tensor * target_consciousness
                },
                "mode": AGIMode.ANALYTICAL
            }
            
            data.append(sample)
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


class MemoryDataset(Dataset):
    """Dataset pour entraîner le système de mémoire."""
    
    def __init__(self, size: int = 400):
        super().__init__()
        
        self.size = size
        self.data = self._generate_memory_data()
        
        logger.info(f"💾 MemoryDataset créé: {size} échantillons mémoire")
    
    def _generate_memory_data(self) -> List[Dict[str, Any]]:
        """Génère des données pour entraîner la mémoire."""
        
        data = []
        
        memory_tasks = [
            "Rappel épisodique",
            "Mémoire sémantique", 
            "Mémoire procédurale",
            "Association",
            "Consolidation"
        ]
        
        for i in range(self.size):
            task = random.choice(memory_tasks)
            
            # Créer une "mémoire" à retenir
            memory_items = []
            num_items = random.randint(1, 5)
            
            for j in range(num_items):
                item = torch.randn(64)  # Embedding mémoire
                memory_items.append(item)
            
            # Input = query pour rappel
            query = torch.randn(32, 256)
            
            # Target = éléments à rappeler
            expected_recall = random.choice(memory_items) if memory_items else torch.zeros(64)
            
            sample = {
                "task": f"Mémoire: {task}",
                "inputs": {"text": query},
                "targets": {
                    "primary_target": query,  # Reconstruction
                    "memory_targets": {
                        "expected_items": memory_items,
                        "expected_count": len(memory_items),
                        "main_recall": expected_recall
                    }
                },
                "mode": AGIMode.LEARNING
            }
            
            data.append(sample)
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def create_sample_dataloaders(
    batch_size: int = 4,
    num_workers: int = 0,
    train_split: float = 0.8
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crée des dataloaders échantillons pour l'entraînement.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Dataset principal multimodal
    full_dataset = SampleMultimodalDataset(size=800)
    
    # Split train/val/test
    train_size = int(len(full_dataset) * train_split)
    val_size = int(len(full_dataset) * 0.15)
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    logger.info(f"📦 Dataloaders créés - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def create_specialized_dataloaders(
    dataset_type: str = "multimodal",
    batch_size: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Crée des dataloaders spécialisés pour phases d'entraînement."""
    
    if dataset_type == "text":
        dataset = TinyTextDataset(size=400)
    elif dataset_type == "consciousness":
        dataset = ConsciousnessDataset(size=200)
    elif dataset_type == "memory":
        dataset = MemoryDataset(size=300)
    else:
        dataset = SampleMultimodalDataset(size=600)
    
    # Split 80/20
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"📊 Dataloaders {dataset_type} créés - Train: {train_size}, Val: {val_size}")
    
    return train_loader, val_loader


def save_sample_data(save_dir: str = "./datasets/samples"):
    """Sauvegarde des échantillons de données pour inspection."""
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Échantillons de chaque type
    datasets_info = {
        "multimodal": SampleMultimodalDataset(size=10),
        "text": TinyTextDataset(size=10),
        "consciousness": ConsciousnessDataset(size=10),
        "memory": MemoryDataset(size=10)
    }
    
    for name, dataset in datasets_info.items():
        samples = []
        for i in range(min(5, len(dataset))):  # 5 échantillons max
            sample = dataset[i]
            
            # Convertir tensors en listes pour JSON
            json_sample = {}
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    json_sample[key] = {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "sample_values": value.flatten()[:10].tolist()  # 10 premières valeurs
                    }
                elif isinstance(value, dict):
                    json_sample[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            json_sample[key][sub_key] = {
                                "shape": list(sub_value.shape),
                                "sample_values": sub_value.flatten()[:5].tolist()
                            }
                        else:
                            json_sample[key][sub_key] = str(sub_value)
                else:
                    json_sample[key] = str(value)
            
            samples.append(json_sample)
        
        # Sauvegarde
        with open(f"{save_dir}/{name}_samples.json", "w") as f:
            json.dump(samples, f, indent=2)
    
    logger.info(f"💾 Échantillons sauvegardés dans {save_dir}")


if __name__ == "__main__":
    # Test des datasets
    print("🧪 Test des datasets échantillons")
    
    # Test multimodal
    dataset = SampleMultimodalDataset(size=50)
    sample = dataset[0]
    print(f"📊 Sample multimodal: {sample['task']}")
    print(f"   Inputs: {list(sample['inputs'].keys())}")
    
    # Test dataloaders
    train_loader, val_loader, test_loader = create_sample_dataloaders(batch_size=2)
    batch = next(iter(train_loader))
    print(f"📦 Batch shape: {len(batch)}")
    
    # Sauvegarde échantillons
    save_sample_data()
    
    print("✅ Tests datasets terminés!")