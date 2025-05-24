"""
Module de projection d'entrée pour NeuroLite.
Remplace les embeddings traditionnels par des projections légères basées sur des
techniques de hachage (MinHash, filtres de Bloom, n-grams) pour réduire la mémoire.
"""

import torch
import torch.nn as nn
import mmh3
import numpy as np
from typing import List, Tuple, Union, Optional
from bitarray import bitarray


class MinHashBloomProjection(nn.Module):
    """
    Couche de projection d'entrée utilisant MinHash et filtres de Bloom.
    Cette approche remplace les grandes tables d'embedding par une projection
    de hachage efficace, comme décrit dans pNLP-Mixer (Fusco et al., 2023).
    """
    
    def __init__(
        self,
        output_dim: int = 256,
        minhash_permutations: int = 128,
        bloom_filter_size: int = 512,
        ngram_range: Tuple[int, int] = (2, 4),
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.minhash_permutations = minhash_permutations
        self.bloom_filter_size = bloom_filter_size
        self.ngram_range = ngram_range
        
        # La projection est une couche linéaire qui projette le vecteur bloom
        # vers la dimension cachée du modèle
        self.projection = nn.Linear(bloom_filter_size, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Génération des seeds pour MinHash (non-entraînable)
        self.minhash_seeds = np.random.randint(
            0, 2**32 - 1, 
            size=minhash_permutations, 
            dtype=np.int64
        )
        
    def _compute_character_ngrams(self, text: str) -> List[str]:
        """Calcule les n-grams de caractères pour un texte donné"""
        min_n, max_n = self.ngram_range
        ngrams = []
        
        # Ajouter des padding spéciaux pour capturer les débuts/fins de mots
        padded_text = f"<{text}>"
        
        # Extraire tous les n-grams entre min_n et max_n
        for n in range(min_n, max_n + 1):
            for i in range(len(padded_text) - n + 1):
                ngrams.append(padded_text[i:i+n])
        
        return ngrams
    
    def _compute_minhash_bloom(self, text: str) -> torch.Tensor:
        """
        Calcule le vecteur MinHash + Bloom pour un texte donné.
        Cette fonction est non-différentiable et non-entraînable.
        """
        ngrams = self._compute_character_ngrams(text)
        
        if not ngrams:
            # Si pas de ngrams (texte vide), retourner un vecteur nul
            return torch.zeros(self.bloom_filter_size)
        
        # Initialiser le filtre de Bloom
        bloom_vector = bitarray(self.bloom_filter_size)
        bloom_vector.setall(0)
        
        # Pour chaque n-gram, calculer MinHash et mettre à jour Bloom
        for ngram in ngrams:
            # Calculer des hash pour chaque permutation
            for seed in self.minhash_seeds:
                # Utiliser mmh3 (MurmurHash3) pour le hachage rapide
                # Convertir numpy.int64 en entier Python standard
                seed_int = int(seed)  # Conversion explicite en int Python
                hash_value = mmh3.hash(ngram, seed_int) % self.bloom_filter_size
                bloom_vector[hash_value] = 1
        
        # Convertir bitarray en tensor
        return torch.tensor([int(bit) for bit in bloom_vector], dtype=torch.float)
    
    def _batch_compute_minhash_bloom(self, texts: List[str]) -> torch.Tensor:
        """Traite un batch de textes pour obtenir leurs vecteurs MinHash+Bloom"""
        batch_vectors = []
        for text in texts:
            vector = self._compute_minhash_bloom(text)
            batch_vectors.append(vector)
            
        # Stack pour former un tensor batch
        return torch.stack(batch_vectors)
    
    def forward(self, input_texts: List[str]) -> torch.Tensor:
        """
        Transforme une liste de textes en représentations vectorielles via
        MinHash + Bloom, puis projette vers la dimension cachée du modèle.
        
        Args:
            input_texts: Liste de chaînes de texte à encoder
            
        Returns:
            Tensor de taille [batch_size, output_dim]
        """
        # Calculer les vecteurs MinHash+Bloom (non-différentiable)
        bloom_vectors = self._batch_compute_minhash_bloom(input_texts)
        
        # Projeter vers la dimension cachée (différentiable)
        hidden = self.projection(bloom_vectors)
        hidden = self.dropout(hidden)
        hidden = self.layer_norm(hidden)
        
        return hidden


class TokenizedMinHashProjection(nn.Module):
    """
    Version adaptée pour traiter des tokens plutôt que du texte brut.
    Cette version fonctionne avec des tokens pré-tokenisés (indices entiers).
    """
    
    def __init__(
        self,
        output_dim: int = 256,
        minhash_permutations: int = 128,
        bloom_filter_size: int = 512,
        vocab_size: int = 30000,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.minhash_permutations = minhash_permutations
        self.bloom_filter_size = bloom_filter_size
        self.vocab_size = vocab_size
        
        # Projection linéaire post-bloom
        self.projection = nn.Linear(bloom_filter_size, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Pré-calculer les hachages pour chaque token possible
        # Cela accélère l'inférence car on calcule les hachages une seule fois
        self._precalculate_token_hashes()
        
    def _precalculate_token_hashes(self):
        """Pré-calcule les hachages MinHash pour tous les tokens du vocabulaire"""
        # Allouer un tensor pour stocker les masques de bits précalculés
        # Shape: [vocab_size, bloom_filter_size]
        self.register_buffer(
            "token_bloom_masks",
            torch.zeros(self.vocab_size, self.bloom_filter_size)
        )
        
        # Pour chaque token possible, calculer son masque bloom
        for token_id in range(self.vocab_size):
            bloom_vector = torch.zeros(self.bloom_filter_size)
            
            # Générer plusieurs hachages pour chaque token
            for seed in range(self.minhash_permutations):
                # Combiner token_id et seed pour la diversité
                hash_input = f"{token_id}_{seed}"
                hash_value = mmh3.hash(hash_input) % self.bloom_filter_size
                bloom_vector[hash_value] = 1
                
            self.token_bloom_masks[token_id] = bloom_vector
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Transforme des indices de tokens en représentations vectorielles.
        
        Args:
            input_ids: Tensor d'indices de tokens [batch_size, seq_length]
            
        Returns:
            Tensor de représentations [batch_size, seq_length, output_dim]
        """
        batch_size, seq_length = input_ids.shape
        
        # Obtenir les masques bloom précalculés pour chaque token
        # Shape: [batch_size, seq_length, bloom_filter_size]
        bloom_vectors = self.token_bloom_masks[input_ids]
        
        # Projeter vers la dimension souhaitée
        # Shape: [batch_size, seq_length, output_dim]
        hidden = self.projection(bloom_vectors)
        hidden = self.dropout(hidden)
        hidden = self.layer_norm(hidden)
        
        return hidden
