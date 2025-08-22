"""
NeuroLite Image Tokenizer
========================

Tokenizer spécialisé pour images avec multiples stratégies:
- Patch Tokenization (ViT-style)
- Pixel-level Tokenization
- Feature-based Tokenization
- Vector Quantization (VQ-VAE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
import time

from ..base_tokenizer import BaseTokenizer, TokenizerConfig, TokenizationResult, ModalityType, TokenizationStrategy

try:
    from PIL import Image
    import torchvision.transforms as transforms
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    Image = None
    transforms = None

logger = logging.getLogger(__name__)

class ImageTokenizer(BaseTokenizer):
    """Tokenizer avancé pour images avec multiples stratégies."""
    
    def __init__(self, config: TokenizerConfig, modality: ModalityType = ModalityType.IMAGE):
        super().__init__(config, modality)
        
        # Configuration spécifique aux images
        image_config = self.modality_config
        self.strategy = TokenizationStrategy(image_config.get('strategy', 'patch'))
        self.patch_size = image_config.get('patch_size', 16)
        self.image_size = image_config.get('image_size', 224)
        self.channels = image_config.get('channels', 3)
        self.embed_dim = image_config.get('embed_dim', 768)
        
        # Preprocessing
        self.normalize = image_config.get('normalize', True)
        self.resize_mode = image_config.get('resize_mode', 'center_crop')
        
        # Initialiser les composants
        self._init_transforms()
        self._init_tokenization_layers()
        
        logger.info(f"ImageTokenizer initialisé avec stratégie {self.strategy.value}")
    
    def _init_transforms(self):
        """Initialise les transformations d'image."""
        if not VISION_AVAILABLE:
            self.transform = None
            return
        
        transform_list = []
        
        # Redimensionnement
        if self.resize_mode == 'center_crop':
            transform_list.extend([
                transforms.Resize(int(self.image_size * 1.2)),
                transforms.CenterCrop(self.image_size)
            ])
        else:
            transform_list.append(transforms.Resize((self.image_size, self.image_size)))
        
        # Conversion en tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalisation
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        
        self.transform = transforms.Compose(transform_list)
    
    def _init_tokenization_layers(self):
        """Initialise les couches de tokenization selon la stratégie."""
        
        if self.strategy == TokenizationStrategy.PATCH:
            self._init_patch_tokenizer()
        elif self.strategy == TokenizationStrategy.PIXEL:
            self._init_pixel_tokenizer()
        elif self.strategy == TokenizationStrategy.FEATURE:
            self._init_feature_tokenizer()
        elif self.strategy == TokenizationStrategy.VQ_VAE:
            self._init_vq_tokenizer()
    
    def _init_patch_tokenizer(self):
        """Initialise la tokenization par patches (ViT-style)."""
        self.patch_embed = nn.Conv2d(
            self.channels, 
            self.embed_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # Calcul du nombre de patches
        self.num_patches = (self.image_size // self.patch_size) ** 2
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, self.embed_dim))
    
    def _init_pixel_tokenizer(self):
        """Initialise la tokenization au niveau pixel."""
        # Quantification des pixels en vocabulaire discret
        self.pixel_vocab_size = 256  # 8-bit pixels
        self.pixel_embed = nn.Embedding(self.pixel_vocab_size, self.embed_dim)
    
    def _init_feature_tokenizer(self):
        """Initialise la tokenization basée sur features."""
        # CNN pour extraction de features
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((14, 14)),  # 14x14 = 196 tokens
            nn.Conv2d(128, self.embed_dim, 1)
        )
    
    def _init_vq_tokenizer(self):
        """Initialise Vector Quantization tokenizer."""
        # Implémentation VQ simplifiée (vq_utils pas encore créé)
        class SimpleVectorQuantizer(nn.Module):
            def __init__(self, codebook_size, embed_dim):
                super().__init__()
                self.codebook_size = codebook_size
                self.embed_dim = embed_dim
                self.codebook = nn.Embedding(codebook_size, embed_dim)
            
            def forward(self, x):
                # VQ simplifié pour démonstration
                flat_x = x.view(-1, self.embed_dim)
                distances = torch.cdist(flat_x, self.codebook.weight)
                indices = torch.argmin(distances, dim=-1)
                quantized = self.codebook(indices)
                commitment_loss = torch.mean((flat_x - quantized.detach()) ** 2)
                return quantized.view_as(x), indices, commitment_loss
        
        # Encoder pour réduire les dimensions
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, self.embed_dim, 3, padding=1)
        )
        
        # Vector Quantizer
        codebook_size = self.modality_config.get('codebook_size', 8192)
        self.vq = SimpleVectorQuantizer(codebook_size, self.embed_dim)
    
    def tokenize(self, data: Any, **kwargs) -> TokenizationResult:
        """
        Tokenise une image.
        
        Args:
            data: Image à tokeniser (PIL, tensor, array, chemin)
            **kwargs: Arguments additionnels
            
        Returns:
            TokenizationResult: Résultat de la tokenization
        """
        start_time = time.time()
        
        try:
            # Preprocessing de l'image
            image_tensor = self._preprocess_image(data)
            
            # Tokenization selon la stratégie
            if self.strategy == TokenizationStrategy.PATCH:
                result = self._tokenize_patches(image_tensor, **kwargs)
            elif self.strategy == TokenizationStrategy.PIXEL:
                result = self._tokenize_pixels(image_tensor, **kwargs)
            elif self.strategy == TokenizationStrategy.FEATURE:
                result = self._tokenize_features(image_tensor, **kwargs)
            elif self.strategy == TokenizationStrategy.VQ_VAE:
                result = self._tokenize_vq(image_tensor, **kwargs)
            else:
                raise ValueError(f"Stratégie {self.strategy.value} non supportée")
            
            # Post-traitement
            result.modality = ModalityType.IMAGE
            result.strategy = self.strategy
            result.original_shape = image_tensor.shape
            result.tokenization_time_ms = (time.time() - start_time) * 1000
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur tokenization image: {e}")
            return TokenizationResult(
                tokens=[],
                modality=ModalityType.IMAGE,
                tokenization_time_ms=(time.time() - start_time) * 1000,
                metadata={'error': str(e)}
            )
    
    def _preprocess_image(self, data: Any) -> torch.Tensor:
        """Préprocesse l'image en tensor normalisé."""
        
        # Conversion selon le type d'entrée
        if isinstance(data, str):
            # Chemin vers fichier
            if VISION_AVAILABLE:
                image = Image.open(data).convert('RGB')
                return self.transform(image).unsqueeze(0)
            else:
                raise ValueError("PIL non disponible pour charger les images")
        
        elif hasattr(data, 'convert'):  # PIL Image
            if VISION_AVAILABLE and self.transform:
                return self.transform(data).unsqueeze(0)
            else:
                # Conversion manuelle
                array = np.array(data)
                tensor = torch.from_numpy(array).float().permute(2, 0, 1) / 255.0
                return tensor.unsqueeze(0)
        
        elif isinstance(data, np.ndarray):
            # Array numpy
            if data.ndim == 3:  # (H, W, C)
                tensor = torch.from_numpy(data).float().permute(2, 0, 1) / 255.0
                return tensor.unsqueeze(0)
            elif data.ndim == 4:  # (B, H, W, C)
                tensor = torch.from_numpy(data).float().permute(0, 3, 1, 2) / 255.0
                return tensor
        
        elif isinstance(data, torch.Tensor):
            # Tensor PyTorch
            if data.ndim == 3:  # (C, H, W)
                return data.unsqueeze(0)
            elif data.ndim == 4:  # (B, C, H, W)
                return data
        
        raise ValueError(f"Format d'image non supporté: {type(data)}")
    
    def _tokenize_patches(self, image: torch.Tensor, **kwargs) -> TokenizationResult:
        """Tokenization par patches (ViT-style)."""
        batch_size = image.shape[0]
        
        # Extraction des patches
        patches = self.patch_embed(image)  # (B, embed_dim, H_patches, W_patches)
        patches = patches.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Ajout des position embeddings
        patches = patches + self.pos_embed
        
        # Les patches sont déjà des embeddings, créer des tokens discrets
        # En quantifiant les embeddings
        tokens = self._quantize_embeddings(patches)
        
        return TokenizationResult(
            tokens=tokens,
            embeddings=patches,
            modality=ModalityType.IMAGE,
            strategy=TokenizationStrategy.PATCH,
            vocab_size=self.get_vocab_size(),
            sequence_length=self.num_patches,
            metadata={
                'patch_size': self.patch_size,
                'num_patches': self.num_patches,
                'embed_dim': self.embed_dim
            }
        )
    
    def _tokenize_pixels(self, image: torch.Tensor, **kwargs) -> TokenizationResult:
        """Tokenization au niveau pixel."""
        # Convertir en pixels 8-bit
        if image.max() <= 1.0:
            image = (image * 255).clamp(0, 255)
        
        pixels = image.long().flatten()  # Tous les pixels
        
        # Sous-échantillonnage si trop de pixels
        max_pixels = kwargs.get('max_pixels', 4096)
        if len(pixels) > max_pixels:
            indices = torch.randperm(len(pixels))[:max_pixels]
            pixels = pixels[indices]
        
        return TokenizationResult(
            tokens=pixels,
            modality=ModalityType.IMAGE,
            strategy=TokenizationStrategy.PIXEL,
            vocab_size=256,
            sequence_length=len(pixels),
            metadata={'max_pixels_used': len(pixels)}
        )
    
    def _tokenize_features(self, image: torch.Tensor, **kwargs) -> TokenizationResult:
        """Tokenization basée sur features CNN."""
        features = self.feature_extractor(image)  # (B, embed_dim, H, W)
        features = features.flatten(2).transpose(1, 2)  # (B, seq_len, embed_dim)
        
        # Quantifier les features
        tokens = self._quantize_embeddings(features)
        
        return TokenizationResult(
            tokens=tokens,
            embeddings=features,
            modality=ModalityType.IMAGE,
            strategy=TokenizationStrategy.FEATURE,
            vocab_size=self.get_vocab_size(),
            sequence_length=features.shape[1],
            metadata={'feature_extraction': 'cnn'}
        )
    
    def _tokenize_vq(self, image: torch.Tensor, **kwargs) -> TokenizationResult:
        """Tokenization avec Vector Quantization."""
        # Encoder l'image
        encoded = self.encoder(image)  # (B, embed_dim, H, W)
        
        # Reshape pour VQ
        B, D, H, W = encoded.shape
        encoded = encoded.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H*W, D)
        
        # Vector Quantization
        quantized, tokens, commitment_loss = self.vq(encoded)
        
        # Reshape back
        tokens = tokens.reshape(B, H * W)
        quantized = quantized.reshape(B, H, W, D).permute(0, 3, 1, 2)
        
        return TokenizationResult(
            tokens=tokens,
            embeddings=quantized,
            modality=ModalityType.IMAGE,
            strategy=TokenizationStrategy.VQ_VAE,
            vocab_size=self.vq.codebook_size,
            sequence_length=H * W,
            metadata={
                'commitment_loss': commitment_loss.item(),
                'spatial_size': (H, W)
            }
        )
    
    def _quantize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Quantifie les embeddings en tokens discrets."""
        # Quantification simple par clustering
        vocab_size = self.get_vocab_size()
        
        # Normaliser les embeddings
        embeddings_norm = F.normalize(embeddings, dim=-1)
        
        # Hasher pour créer des tokens discrets
        # (implémentation simplifiée, remplacer par k-means en production)
        tokens = torch.sum(embeddings_norm * 1000, dim=-1).long() % vocab_size
        
        return tokens
    
    def detokenize(self, tokens: Union[torch.Tensor, np.ndarray, List[int]], **kwargs) -> torch.Tensor:
        """
        Reconstruit une image à partir des tokens.
        
        Args:
            tokens: Tokens à détokeniser
            **kwargs: Arguments additionnels
            
        Returns:
            torch.Tensor: Image reconstruite
        """
        try:
            if isinstance(tokens, (list, np.ndarray)):
                tokens = torch.tensor(tokens)
            
            if self.strategy == TokenizationStrategy.PATCH:
                return self._detokenize_patches(tokens, **kwargs)
            elif self.strategy == TokenizationStrategy.PIXEL:
                return self._detokenize_pixels(tokens, **kwargs)
            elif self.strategy == TokenizationStrategy.VQ_VAE:
                return self._detokenize_vq(tokens, **kwargs)
            else:
                # Reconstruction approximative
                return self._approximate_reconstruction(tokens, **kwargs)
                
        except Exception as e:
            logger.error(f"Erreur détokenization image: {e}")
            return torch.zeros(1, 3, self.image_size, self.image_size)
    
    def _detokenize_patches(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """Reconstruit depuis des tokens de patches."""
        # Conversion tokens -> embeddings (approximative)
        embeddings = self.get_embeddings(tokens)
        
        # Reshape en patches
        B = embeddings.shape[0]
        patches_per_side = int(np.sqrt(self.num_patches))
        
        # Reconstruction approximative par déconvolution
        patches = embeddings.reshape(B, patches_per_side, patches_per_side, self.embed_dim)
        patches = patches.permute(0, 3, 1, 2)  # (B, embed_dim, H, W)
        
        # Déconvolution pour reconstruire l'image
        reconstruction = F.conv_transpose2d(
            patches, 
            self.patch_embed.weight, 
            stride=self.patch_size
        )
        
        return reconstruction
    
    def _detokenize_pixels(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """Reconstruit depuis des tokens de pixels."""
        # Reconstruction très simplifiée
        target_size = kwargs.get('target_size', (3, self.image_size, self.image_size))
        
        # Répéter les tokens pour remplir l'image
        reconstruction = tokens.float().unsqueeze(0).unsqueeze(0)
        reconstruction = F.interpolate(reconstruction, size=target_size[1:], mode='nearest')
        reconstruction = reconstruction.repeat(1, target_size[0], 1, 1)
        
        return reconstruction / 255.0
    
    def _detokenize_vq(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """Reconstruit depuis des tokens VQ."""
        # Utiliser le codebook VQ pour reconstruire
        quantized = self.vq.get_codebook_entry(tokens)
        
        # Decoder (pas implémenté dans cet exemple)
        # En pratique, il faudrait un decoder symétrique à l'encoder
        return torch.randn(1, 3, self.image_size, self.image_size)
    
    def _approximate_reconstruction(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """Reconstruction approximative générique."""
        return torch.randn(1, 3, self.image_size, self.image_size)
    
    def get_vocab_size(self) -> int:
        """Retourne la taille du vocabulaire."""
        if self.strategy == TokenizationStrategy.PIXEL:
            return 256
        elif self.strategy == TokenizationStrategy.VQ_VAE:
            return getattr(self.vq, 'codebook_size', 8192)
        else:
            return self.config.vocab_size
    
    def get_supported_strategies(self) -> List[TokenizationStrategy]:
        """Retourne les stratégies supportées."""
        return [
            TokenizationStrategy.PATCH,
            TokenizationStrategy.PIXEL,
            TokenizationStrategy.FEATURE,
            TokenizationStrategy.VQ_VAE
        ]