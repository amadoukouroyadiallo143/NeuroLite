"""
Gestionnaire de données pour NeuroLite.
Charge et prépare les données d'entraînement, validation et test pour la génération de texte.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
import re
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Fonction collate définie au niveau du module pour permettre la sérialisation
def dataset_collate_fn(examples):
    """Fonction de collation pour les datasets PyTorch.
    
    Convertit les listes d'éléments en un batch structuré avec des tenseurs.
    Définie au niveau du module pour permettre la sérialisation avec multiprocessing.
    """
    # Vérifier que tous les exemples sont des dictionnaires
    if len(examples) == 0 or not all(isinstance(ex, dict) for ex in examples):
        raise ValueError(f"Batch invalide: {examples}")
    
    # Initialiser le batch
    batch = {}
    
    # Traiter les tenseurs
    for key in ["input_ids", "target_ids"]:
        if key in examples[0]:
            # S'assurer que tous les éléments sont des tenseurs
            tensors = []
            for ex in examples:
                if isinstance(ex[key], torch.Tensor):
                    tensors.append(ex[key])
                else:
                    tensors.append(torch.tensor(ex[key], dtype=torch.long))
            batch[key] = torch.stack(tensors)
    
    # Traiter les autres champs
    for key in examples[0].keys():
        if key not in batch and key in examples[0]:
            batch[key] = [ex[key] for ex in examples]
    
    return batch

class TextTokenizer:
    """
    Tokenizer simple pour le texte.
    Convertit le texte en tokens pour l'entraînement de modèles de génération.
    Compatible avec le modèle NeuroLite unifié.
    """
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_freqs = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
        
        # Initialiser avec les tokens spéciaux
        for token, idx in self.special_tokens.items():
            self.word_to_idx[token] = idx
            self.idx_to_word[idx] = token
        
        self.vocab_initialized = False
    
    def build_vocab(self, texts: List[str]):
        """Construit le vocabulaire à partir d'une liste de textes"""
        # Compter les fréquences des mots
        for text in texts:
            for word in self._tokenize(text):
                if word not in self.word_freqs:
                    self.word_freqs[word] = 0
                self.word_freqs[word] += 1
        
        # Trier les mots par fréquence
        sorted_words = sorted(self.word_freqs.items(), key=lambda x: x[1], reverse=True)
        
        # Prendre les vocab_size - len(special_tokens) mots les plus fréquents
        vocab_limit = self.vocab_size - len(self.special_tokens)
        
        # Ajouter les mots au vocabulaire
        for i, (word, _) in enumerate(sorted_words[:vocab_limit]):
            idx = i + len(self.special_tokens)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        self.vocab_initialized = True
        logger.info(f"Vocabulaire construit avec {len(self.word_to_idx)} mots")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenise un texte en mots"""
        # Supprimer les caractères spéciaux et diviser en mots
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode un texte en indices de tokens"""
        if not self.vocab_initialized:
            raise ValueError("Le vocabulaire n'a pas été initialisé. Appelez build_vocab d'abord.")
        
        tokens = self._tokenize(text)
        ids = []
        
        if add_special_tokens:
            ids.append(self.special_tokens['<BOS>'])
        
        for token in tokens:
            if token in self.word_to_idx:
                ids.append(self.word_to_idx[token])
            else:
                ids.append(self.special_tokens['<UNK>'])
        
        if add_special_tokens:
            ids.append(self.special_tokens['<EOS>'])
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Décode une liste d'indices en texte"""
        tokens = []
        
        for idx in ids:
            if idx in self.idx_to_word:
                token = self.idx_to_word[idx]
                
                # Ignorer les tokens spéciaux si demandé
                if skip_special_tokens and token in self.special_tokens:
                    continue
                    
                tokens.append(token)
            else:
                tokens.append(self.idx_to_word[self.special_tokens['<UNK>']])
        
        return ' '.join(tokens)


class WikiTextDataset(Dataset):
    """
    Dataset pour les données WikiText.
    Charge les textes à partir des métadonnées et les prépare pour la génération de texte.
    Optimisé pour le modèle NeuroLite unifié avec task_type="generation".
    """
    
    def __init__(
        self, 
        data_dir: str,
        tokenizer: TextTokenizer,
        split: str = "train",
        seq_length: int = 128,
        stride: int = 64,
        max_samples: int = None
    ):
        """
        Initialise le dataset WikiText pour la génération de texte.
        
        Args:
            data_dir: Chemin vers le répertoire de données Wikitext
            tokenizer: Tokenizer pour convertir le texte en tokens
            split: Division des données ('train', 'val', 'test')
            seq_length: Longueur des séquences pour l'entraînement
            stride: Pas pour la fenêtre glissante lors de la création des séquences
            max_samples: Nombre maximum d'échantillons à charger (pour les tests). Si None, charge tous les échantillons.
        """
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride
        self.max_samples = max_samples
        
        # Charger les métadonnées
        metadata_path = os.path.join(data_dir, f"{split}_metadata.json")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.raw_samples = self.metadata["samples"]
        
        # Limiter le nombre d'échantillons si max_samples est spécifié
        if self.max_samples is not None and self.max_samples > 0 and self.max_samples < len(self.raw_samples):
            self.raw_samples = self.raw_samples[:self.max_samples]
            logger.info(f"Chargé {len(self.raw_samples)} échantillons bruts sur un total disponible de {len(self.metadata['samples'])} pour la division {split}")
        else:
            logger.info(f"Chargé {len(self.raw_samples)} échantillons bruts pour la division {split}")
        
        # Charger tous les textes
        self.all_texts = []
        for sample in self.raw_samples:
            # Correction du chemin: remplacer les backslashes par des slashes et construire le chemin absolu
            relative_path = sample["modalities"]["text"]["path"].replace('\\', '/')
            # Dans le cas d'un chemin relatif commençant par 'text/', préfixer avec le split
            if relative_path.startswith('text/'):
                relative_path = f"{split}/{relative_path}"
            elif not relative_path.startswith(f"{split}/"):
                relative_path = f"{split}/{relative_path}"
            
            text_path = os.path.join(data_dir, relative_path)
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    self.all_texts.append(f.read())
            except FileNotFoundError:
                # Essayer une alternative si le premier chemin ne fonctionne pas
                alternative_path = os.path.join(data_dir, split, "text", os.path.basename(relative_path))
                with open(alternative_path, 'r', encoding='utf-8') as f:
                    self.all_texts.append(f.read())
        
        # Construire le vocabulaire si c'est le split d'entraînement
        if split == "train" and not tokenizer.vocab_initialized:
            tokenizer.build_vocab(self.all_texts)
        
        # Créer les séquences
        self.sequences = self._create_sequences()
        logger.info(f"Créé {len(self.sequences)} séquences pour la division {split}")
    
    def _create_sequences(self) -> List[Dict]:
        """
        Crée des séquences à partir des textes pour l'entraînement de génération.
        Utilise une fenêtre glissante pour créer des séquences qui se chevauchent.
        Gère également les textes plus courts en ajoutant du padding si nécessaire.
        """
        sequences = []
        pad_token_id = self.tokenizer.special_tokens['<PAD>']
        
        for i, text in enumerate(self.all_texts):
            # Encoder le texte complet
            token_ids = self.tokenizer.encode(text)
            
            # Vérifier si le texte est trop court pour la longueur de séquence demandée
            if len(token_ids) < self.seq_length:
                # Si c'est le cas et qu'il a une taille minimale raisonnable (au moins 20% de seq_length)
                if len(token_ids) >= max(50, int(self.seq_length * 0.2)):
                    # Ajouter du padding pour atteindre seq_length
                    padded_seq = token_ids + [pad_token_id] * (self.seq_length - len(token_ids))
                    sequences.append({
                        "input_ids": padded_seq,
                        "target_ids": padded_seq,
                        "sample_id": self.raw_samples[i]["id"]
                    })
                continue  # Passer au texte suivant
                
            # Créer des séquences avec une fenêtre glissante pour les textes assez longs
            for start_idx in range(0, len(token_ids) - self.seq_length + 1, self.stride):
                end_idx = start_idx + self.seq_length
                
                # Pour le modèle NeuroLite unifié en mode génération, nous utilisons la même séquence
                # pour input_ids et labels (décalés de 1) - le modèle gère le décalage en interne
                full_seq = token_ids[start_idx:end_idx]
                
                # S'assurer que la séquence a la bonne longueur
                if len(full_seq) == self.seq_length:
                    sequences.append({
                        "input_ids": full_seq,
                        "target_ids": full_seq,
                        "sample_id": self.raw_samples[i]["id"]
                    })
        
        return sequences
    
    def __len__(self) -> int:
        """Renvoie le nombre de séquences dans le dataset"""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Récupère une séquence du dataset par son index.
        
        Args:
            idx: Index de la séquence à récupérer
            
        Returns:
            Dictionnaire contenant les séquences d'entrée et cible sous forme de tenseurs PyTorch
        """
        sequence = self.sequences[idx]
        
        # Convertir en tenseurs PyTorch
        input_ids = torch.tensor(sequence["input_ids"], dtype=torch.long)
        target_ids = torch.tensor(sequence["target_ids"], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "sample_id": sequence["sample_id"]
        }


class MultimodalDataset(Dataset):
    """
    Dataset multimodal pour NeuroLite.
    Supporte le chargement de données de différentes modalités (texte, image, audio).
    """
    
    def __init__(
        self, 
        data_dir: str,
        split: str = "train",
        modalities: List[str] = ["text"],
        max_text_length: int = 512,
        transform_image = None,
        transform_audio = None
    ):
        """
        Initialise le dataset multimodal.
        
        Args:
            data_dir: Chemin vers le répertoire de données
            split: Division des données ('train', 'val', 'test')
            modalities: Liste des modalités à charger ('text', 'image', 'audio')
            max_text_length: Longueur maximale des séquences de texte
            transform_image: Transformations à appliquer aux images
            transform_audio: Transformations à appliquer aux données audio
        """
        self.data_dir = data_dir
        self.split = split
        self.modalities = modalities
        self.max_text_length = max_text_length
        self.transform_image = transform_image
        self.transform_audio = transform_audio
        
        # Charger les métadonnées
        metadata_path = os.path.join(data_dir, f"{split}_metadata.json")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.samples = self.metadata["samples"]
        logger.info(f"Chargé {len(self.samples)} échantillons multimodaux pour la division {split}")
        
        # Vérifier les modalités disponibles
        dataset_modalities = []
        if os.path.exists(os.path.join(data_dir, "metadata.json")):
            with open(os.path.join(data_dir, "metadata.json"), 'r', encoding='utf-8') as f:
                global_metadata = json.load(f)
                if "modalities" in global_metadata:
                    dataset_modalities = global_metadata["modalities"]
        
        for modality in modalities:
            if modality not in dataset_modalities and dataset_modalities:
                logger.warning(f"Modalité '{modality}' non trouvée dans les métadonnées du dataset")
    
    def __len__(self) -> int:
        """Renvoie le nombre d'échantillons dans le dataset"""
        return len(self.samples)
    
    def load_text(self, sample_metadata: Dict) -> str:
        """Charge les données texte d'un échantillon"""
        if "text" not in sample_metadata["modalities"]:
            return ""
            
        text_path = os.path.join(self.data_dir, sample_metadata["modalities"]["text"]["path"])
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tronquer à la longueur maximale si nécessaire
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]
            
        return text
    
    def load_image(self, sample_metadata: Dict) -> Optional[torch.Tensor]:
        """Charge les données image d'un échantillon"""
        if "image" not in sample_metadata["modalities"]:
            return None
            
        try:
            from PIL import Image
            import numpy as np
            
            image_path = os.path.join(self.data_dir, sample_metadata["modalities"]["image"]["path"])
            image = Image.open(image_path).convert('RGB')
            
            if self.transform_image:
                image = self.transform_image(image)
            else:
                # Transformation par défaut
                image = np.array(image)
                image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
                
            return image
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'image: {e}")
            return None
    
    def load_audio(self, sample_metadata: Dict) -> Optional[torch.Tensor]:
        """Charge les données audio d'un échantillon"""
        if "audio" not in sample_metadata["modalities"]:
            return None
            
        try:
            import torchaudio
            
            audio_path = os.path.join(self.data_dir, sample_metadata["modalities"]["audio"]["path"])
            waveform, sample_rate = torchaudio.load(audio_path)
            
            if self.transform_audio:
                waveform = self.transform_audio(waveform)
                
            return waveform
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'audio: {e}")
            return None
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Récupère un échantillon multimodal du dataset par son index.
        
        Args:
            idx: Index de l'échantillon à récupérer
            
        Returns:
            Dictionnaire contenant les données multimodales de l'échantillon
        """
        sample_metadata = self.samples[idx]
        sample_id = sample_metadata["id"]
        
        # Initialiser le dictionnaire de sortie
        sample = {"id": sample_id}
        
        # Charger chaque modalité demandée
        if "text" in self.modalities:
            sample["text"] = self.load_text(sample_metadata)
            
        if "image" in self.modalities:
            sample["image"] = self.load_image(sample_metadata)
            
        if "audio" in self.modalities:
            sample["audio"] = self.load_audio(sample_metadata)
            
        return sample


class DataManager:
    """
    Gestionnaire de données pour NeuroLite.
    Gère le chargement et la préparation des données pour l'entraînement de génération de texte.
    Compatible avec le modèle NeuroLite unifié (task_type="generation").
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        seq_length: int = 128,
        stride: int = 64,
        vocab_size: int = 10000,
        max_samples: int = None
    ):
        """
        Initialise le gestionnaire de données pour la génération de texte.
        
        Args:
            data_dir: Chemin vers le répertoire de données
            batch_size: Taille des batchs pour les DataLoaders
            num_workers: Nombre de workers pour le chargement parallèle
            seq_length: Longueur des séquences pour l'entraînement
            stride: Pas pour la fenêtre glissante lors de la création des séquences
            vocab_size: Taille du vocabulaire
            max_samples: Nombre maximum d'échantillons à charger par division (pour les tests). Si None, charge tous les échantillons.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_length = seq_length
        self.stride = stride
        self.max_samples = max_samples
        
        # Charger les métadonnées globales
        metadata_path = os.path.join(data_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"Métadonnées globales chargées: {self.metadata['name']}")
        else:
            self.metadata = {"name": os.path.basename(data_dir), "modalities": ["text"]}
            logger.warning("Métadonnées globales non trouvées, utilisation de valeurs par défaut")
        
        # Initialiser le tokenizer
        self.tokenizer = TextTokenizer(vocab_size=vocab_size)
    
    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Crée et renvoie les DataLoaders pour l'entraînement, la validation et le test.
        
        Returns:
            Dictionnaire contenant les DataLoaders pour chaque division
        """
        datasets = {}
        dataloaders = {}
        
        # Réactiver le multiprocessing puisque la fonction collate est maintenant au niveau module
        effective_num_workers = self.num_workers
        
        # Créer un dataset pour chaque division (train, val, test)
        for split in ["train", "val", "test"]:
            try:
                # Dataset texte pour la génération (WikiText)
                dataset = WikiTextDataset(
                    data_dir=self.data_dir,
                    tokenizer=self.tokenizer,
                    split=split,
                    seq_length=self.seq_length,
                    stride=self.stride,
                    max_samples=self.max_samples
                )
                
                datasets[split] = dataset
                
                # Créer le DataLoader avec la fonction collate globale
                dataloaders[split] = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=(split == "train"),
                    num_workers=effective_num_workers,  # Éviter le multiprocessing
                    pin_memory=torch.cuda.is_available(),
                    collate_fn=dataset_collate_fn  # Utiliser la fonction globale
                )
                
                logger.info(f"DataLoader créé pour {split} avec {len(dataset)} échantillons")
            except Exception as e:
                logger.error(f"Erreur lors de la création du dataset {split}: {e}")
        
        return dataloaders
    
    def get_train_val_test_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Renvoie les DataLoaders pour l'entraînement, la validation et le test.
        
        Returns:
            Tuple contenant (train_loader, val_loader, test_loader)
        """
        dataloaders = self.get_dataloaders()
        
        train_loader = dataloaders.get("train", None)
        val_loader = dataloaders.get("val", None)
        test_loader = dataloaders.get("test", None)
        
        return train_loader, val_loader, test_loader
