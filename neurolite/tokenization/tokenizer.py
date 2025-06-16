"""
Tokenizer multimodal unifié pour l'architecture NeuroLite.

Ce module implémente un tokenizer qui :
1. Projette les différentes modalités (texte, image, etc.) dans un espace commun.
2. Utilise un `NeuralCompressor` pour discrétiser la représentation commune en tokens.
"""

import os
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, Any, Optional, List
from collections import Counter
from datasets import Dataset
from tqdm import tqdm

from neurolite.Configs.config import TokenizerConfig, NeuroLiteConfig
from neurolite.multimodal.multimodal import MultiModalEncoders
from .compressor import NeuralCompressor
from .projectors import CrossModalProjector
from .losses import MultimodalContrastiveLoss
from .quantizers import ResidualVQ
from .tokenizers.bpe_tokenizer import BPETokenizer

class NeuroLiteTokenizer(nn.Module):
    """
    Le Tokenizer central de NeuroLite. Il gère l'encodage de toutes les modalités,
    la projection, la compression et la quantification en tokens discrets.
    """
    def __init__(self, config: TokenizerConfig, neurolite_config: "NeuroLiteConfig"):
        """
        Initialise le NeuroLiteTokenizer.
        """
        super().__init__()
        self.config = config
        self.neurolite_config = neurolite_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer_type = config.tokenizer_type
        self.text_tokenizer = None # Renommé pour plus de clarté

        # Initialisation du tokenizer sous-jacent pour le texte
        if self.tokenizer_type == 'bpe':
            self.text_tokenizer = BPETokenizer(
                vocab_size=getattr(config, 'vocab_size', 50000), 
                special_tokens=['[PAD]', '[UNK]', '[BOS]', '[EOS]']
            )
        else:
            raise ValueError(f"Tokenizer type '{self.tokenizer_type}' is not supported. Use 'bpe'.")

        # --- Vocabulaire propriétaire ---
        self.special_tokens = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.char_to_id = self.special_tokens.copy()
        self.id_to_char = {v: k for k, v in self.special_tokens.items()}
        self.vocab_size = len(self.special_tokens)

        self.pad_token_id = self.special_tokens['<PAD>']
        self.unk_token_id = self.special_tokens['<UNK>']
        self.bos_token_id = self.special_tokens['<BOS>']
        self.eos_token_id = self.special_tokens['<EOS>']

        # Modules principaux du tokenizer
        self.encoders = MultiModalEncoders(neurolite_config.model_config).to(self.device)
        
        modality_dims = { name: encoder.config.output_dim for name, encoder in self.encoders.encoders.items() }

        self.projector = CrossModalProjector(
            input_dims=modality_dims,
            output_dim=config.hidden_size,
            dropout_rate=config.projection_dropout
        ).to(self.device)
        
        self.compressor = NeuralCompressor(
            input_dim=config.hidden_size,
            bottleneck_dim=config.compressor_bottleneck_dim,
            num_quantizers=config.num_quantizers,
            codebook_size=config.codebook_size,
            commitment_weight=config.commitment_weight,
            ema_decay=config.ema_decay
        ).to(self.device)
        
        self.alignment_loss_fn = MultimodalContrastiveLoss(temperature=config.contrastive_temperature).to(self.device)

    def build_vocab(self, dataset: Dataset, text_column: str = 'text', chat_format: bool = False):
        """
        Entraîne le tokenizer de texte (BPE) sur un dataset.
        """
        print(f"Training '{self.tokenizer_type}' tokenizer on the dataset...")
        
        def get_text_iterator():
            progress_bar = tqdm(dataset, desc="[1/2] Preparing corpus")
            if chat_format:
                for item in progress_bar:
                    yield "".join(msg['content'] for msg in item['messages'])
            else:
                for item in progress_bar:
                    yield item[text_column]
        
        corpus = list(get_text_iterator())
        
        if self.tokenizer_type == 'bpe':
            print("\n[2/2] Starting BPE training... (this may take a while)")
            self.text_tokenizer.train(corpus)
            self.vocab_size = self.text_tokenizer.get_vocab_size()
            print(f"BPE Vocabulary built. Total size: {self.vocab_size} tokens.")

            # --- CORRECTION CRUCIALE ---
            # Mettre à jour la configuration et redimensionner la couche d'embedding
            self.config.vocab_size = self.vocab_size
            
            # Accéder au ModuleDict correctement
            if 'text' in self.encoders.encoders:
                text_encoder = self.encoders.encoders['text']
                text_encoder.resize_token_embeddings(self.vocab_size)
                print(f"Text encoder token embedding layer resized to {self.vocab_size}.")
            # --- FIN DE LA CORRECTION ---
        else:
            print(f"Tokenizer type '{self.tokenizer_type}' does not require explicit vocabulary building.")

    def encode_text(self, texts: List[str], max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Encode une liste de textes en utilisant le tokenizer de texte sous-jacent (BPE).
        """
        if not self.text_tokenizer:
            raise RuntimeError("Text tokenizer has not been initialized.")

        # Utiliser notre nouveau tokenizer BPE
        batch_input_ids = [self.text_tokenizer.encode(text) for text in texts]

        # Padding et Truncation manuels pour le moment
        # Une implémentation plus avancée gérerait cela plus élégamment
        max_len_in_batch = max(len(ids) for ids in batch_input_ids)
        if max_length:
            max_len_in_batch = min(max_len_in_batch, max_length)

        # Assumons que [PAD] est le token 0
        pad_token_id = self.text_tokenizer.vocab.get('[PAD]', 0)

        padded_batch = []
        attention_masks = []
        for ids in batch_input_ids:
            # Truncation
            truncated_ids = ids[:max_len_in_batch]
            
            # Padding
            padded_ids = truncated_ids + [pad_token_id] * (max_len_in_batch - len(truncated_ids))
            mask = [1] * len(truncated_ids) + [0] * (max_len_in_batch - len(truncated_ids))
            
            padded_batch.append(padded_ids)
            attention_masks.append(mask)

        return {
            "input_ids": torch.tensor(padded_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long)
        }

    def decode_text(self, token_ids: torch.Tensor) -> List[str]:
        """
        Décode des IDs de token en texte.
        """
        if not self.text_tokenizer:
            raise RuntimeError("Text tokenizer has not been initialized.")
        
        decoded_texts = []
        for ids in token_ids.cpu().numpy():
            # Filtrer les tokens de padding avant de décoder
            pad_token_id = self.text_tokenizer.vocab.get('[PAD]', 0)
            valid_ids = [id for id in ids if id != pad_token_id]
            decoded_texts.append(self.text_tokenizer.decode(valid_ids))
        return decoded_texts

    @torch.no_grad()
    def tokenize(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Tokenize et compresse les données multimodales en une séquence d'indices discrets.
        """
        processed_inputs = {}
        max_seq_length = self.neurolite_config.model_config.max_seq_length

        for modality, data in inputs.items():
            if modality == 'text':
                if isinstance(data, dict) and 'input_ids' in data:
                    # Les données sont déjà tokenisées (contiennent input_ids), on les utilise directement
                    processed_inputs[modality] = data
                elif isinstance(data, list):
                     # Les données sont du texte brut, on les encode
                     processed_inputs[modality] = self.encode_text(data, max_length=max_seq_length)
                else:
                    raise TypeError(f"L'entrée texte doit être une liste de chaînes ou un dict tokenisé, mais a reçu {type(data)}")
            else:
                processed_inputs[modality] = data

        encoded_features = self.encoders(processed_inputs)
        projected_features = self.projector(encoded_features)
        indices = self.compressor.compress(projected_features)
        return indices

    def forward(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Passe forward complète du tokenizer pour l'entraînement.
        """
        encoded_features = self.encoders(inputs)
        projected_features = self.projector(encoded_features)
        compressor_output = self.compressor(projected_features)
        reconstructed_features = compressor_output['reconstructed']
        commitment_loss = compressor_output['commitment_loss']
        
        reconstruction_loss = F.mse_loss(reconstructed_features, projected_features.detach())
        alignment_loss = self.alignment_loss_fn(encoded_features)
        
        total_loss = (
            self.config.reconstruction_weight * reconstruction_loss +
            self.config.commitment_weight * commitment_loss +
            self.config.alignment_weight * alignment_loss
        )
        
        return {
            'loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'commitment_loss': commitment_loss,
            'alignment_loss': alignment_loss,
            'indices': compressor_output['indices'],
            'quantized': compressor_output['quantized']
        }
        
    def save_pretrained(self, save_directory: str):
        """
        Sauvegarde la configuration, le vocabulaire et les poids du tokenizer.
        """
        os.makedirs(save_directory, exist_ok=True)
        
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        vocab_path = os.path.join(save_directory, "vocab.json")
        with open(vocab_path, 'w') as f:
            json.dump(self.char_to_id, f, indent=2)
            
        # On ne sauvegarde plus le state_dict du nn.Module.
        # Ce rôle est laissé au modèle principal.
        # weights_path = os.path.join(save_directory, "tokenizer.pt")
        # torch.save(self.state_dict(), weights_path)

        # Sauvegarder la configuration spécifique au tokenizer de texte
        if self.text_tokenizer and hasattr(self.text_tokenizer, 'save'):
            text_tokenizer_path = os.path.join(save_directory, 'text_tokenizer')
            self.text_tokenizer.save(text_tokenizer_path)

    @classmethod
    def from_pretrained(cls, save_directory: str, neurolite_config: "NeuroLiteConfig"):
        """
        Charge la configuration, le vocabulaire et les poids du tokenizer.
        """
        # 1. Charger la config du tokenizer pour connaître la taille du vocabulaire cible.
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Fichier de configuration 'tokenizer_config.json' non trouvé dans {save_directory}")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        tokenizer_config = TokenizerConfig.from_dict(config_dict)

        # 2. Mettre à jour la config globale AVANT l'initialisation pour que les modules aient la bonne taille.
        if hasattr(tokenizer_config, 'vocab_size') and tokenizer_config.vocab_size is not None:
            neurolite_config.model_config.mm_text_encoder_config.vocab_size = tokenizer_config.vocab_size
            if neurolite_config.model_config.mm_text_decoder_config:
                neurolite_config.model_config.mm_text_decoder_config.vocab_size = tokenizer_config.vocab_size
        
        # 3. Initialiser le tokenizer avec la configuration mise à jour.
        instance = cls(tokenizer_config, neurolite_config)
        
        # 4. Charger le vocabulaire du tokenizer BPE.
        text_tokenizer_path = os.path.join(save_directory, 'text_tokenizer')
        if os.path.exists(text_tokenizer_path):
            if instance.tokenizer_type == 'bpe':
                instance.text_tokenizer = BPETokenizer.load(text_tokenizer_path)
                
                # S'assurer que la taille du vocabulaire et la couche d'embedding sont parfaitement synchronisées.
                final_vocab_size = instance.text_tokenizer.get_vocab_size()
                instance.vocab_size = final_vocab_size
                instance.config.vocab_size = final_vocab_size
                
                if 'text' in instance.encoders.encoders:
                    text_encoder = instance.encoders.encoders['text']
                    text_encoder.resize_token_embeddings(final_vocab_size)
                    print(f"Tokenizer BPE chargé. Couche d'embedding du modèle synchronisée à {final_vocab_size} tokens.")
        else:
            print("Attention: Aucun dossier 'text_tokenizer' trouvé. Le tokenizer BPE n'a pas été chargé.")

        return instance

    def decode(self, indices: List[torch.Tensor]) -> torch.Tensor:
        """
        Décode une liste de codes discrets en une représentation continue.
        """
        return self.compressor.decompress(indices)
