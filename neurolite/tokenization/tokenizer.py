"""
Tokenizer multimodal avancé pour l'architecture NeuroLite.

Ce module implémente un tokenizer multimodal sophistiqué qui étend les capacités
des projections multimodales existantes avec une tokenization hiérarchique
et une architecture à plusieurs niveaux pour une représentation unifiée.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Any, TypeVar

from ..Configs.config import TokenizerConfig, BaseConfig
from ..multimodal.encoders import (
    TextEncoder, ImageEncoder, AudioEncoder,
    VideoEncoder, GraphEncoder
)
from .quantizers import VectorQuantizer, ResidualVQ
from .hierarchical import HierarchicalTokenizer

# Type variable pour les configurations
T = TypeVar('T', bound='BaseConfig')


class NeuroLiteTokenizer(nn.Module):
    """
    Tokenizer multimodal avancé pour NeuroLite.
    
    Implémente une tokenization hiérarchique avec des codebooks multiples
    pour une représentation unifiée des différentes modalités.
    """
    
    def __init__(self, config: TokenizerConfig):
        """
        Initialise le tokenizer multimodal NeuroLite.
        
        Args:
            config: Configuration du tokenizer
        """
        super().__init__()
        self.config = config
        
        # Encodeurs spécifiques à chaque modalité
        self.encoders = nn.ModuleDict({
            'text': TextEncoder(config.text_encoder_config),
            'image': ImageEncoder(config.vision_encoder_config),
            'audio': AudioEncoder(config.audio_encoder_config),
            'video': VideoEncoder(config.video_encoder_config),
            'graph': GraphEncoder(config.graph_encoder_config)
        })
        
        # Projection des modalités vers l'espace latent commun
        self.modality_projections = nn.ModuleDict({
            mod: nn.Linear(
                config.modality_dims[mod],
                config.hidden_size
            )
            for mod in config.modality_dims
        })
        
        # Attention intermodale pour l'alignement
        self.cross_modal_attention = CrossModalAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_alignment_heads,
            dropout_rate=config.alignment_dropout
        )
        
        # Projection post-attention pour normalisation
        self.post_attention_norm = nn.LayerNorm(config.hidden_size)
        self.post_attention_projection = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Codebook sémantique pour la compréhension (granularité plus grossière)
        self.semantic_codebook = VectorQuantizer(
            n_embeddings=config.semantic_vocab_size,
            embedding_dim=config.hidden_size,
            commitment_cost=config.commitment_cost,
            use_ema_updates=True,
            ema_decay=config.ema_decay
        )
        
        # Codebook détaillé pour la génération (granularité plus fine)
        self.detail_codebook = VectorQuantizer(
            n_embeddings=config.detail_vocab_size,
            embedding_dim=config.hidden_size,
            commitment_cost=config.commitment_cost * 1.5,  # Pénalité plus forte
            use_ema_updates=True,
            ema_decay=config.ema_decay
        )
        
        # Couche de raffinement entre codebooks
        self.codebook_refinement = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size)
        )
        
        # Quantificateur vectoriel résiduel pour compression avancée
        self.residual_quantizer = ResidualVQ(
            dim=config.hidden_size,
            num_quantizers=config.num_quantizers,
            codebook_size=config.codebook_size,
            shared_codebook=config.shared_codebook,
            commitment_weight=config.commitment_weight,
            ema_decay=config.ema_decay
        )
        
        # Projecteur de sortie
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)
    
    def encode_multimodal(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Encode les entrées multimodales en représentations intermédiaires.
        
        Args:
            inputs: Dictionnaire d'entrées multimodales avec clés 'text', 'image', etc.
            
        Returns:
            Dictionnaire des représentations encodées par modalité
        """
        encoded_features = {}
        
        # Traiter chaque modalité avec son encodeur dédié
        for modality, encoder in self.encoders.items():
            if modality in inputs and inputs[modality] is not None:
                # Encoder la modalité
                features = encoder(inputs[modality])
                
                # Projeter dans l'espace latent commun
                if modality in self.modality_projections:
                    features = self.modality_projections[modality](features)
                
                encoded_features[modality] = features
        
        return encoded_features
    
    def align_modalities(self, encoded_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Aligne les différentes modalités dans un espace commun.
        
        Args:
            encoded_features: Dictionnaire des représentations encodées par modalité
            
        Returns:
            Représentation alignée unifiée
        """
        modalities = list(encoded_features.keys())
        
        if not modalities:
            raise ValueError("Aucune modalité valide fournie pour l'alignement")
            
        if len(modalities) == 1:
            # Une seule modalité, pas besoin d'alignement complexe
            return list(encoded_features.values())[0]
        
        # Pour plusieurs modalités, utiliser une attention croisée
        batch_size = next(iter(encoded_features.values())).shape[0]
        device = next(iter(encoded_features.values())).device
        
        # Initialiser avec la moyenne des caractéristiques
        combined_features = torch.stack(list(encoded_features.values())).mean(dim=0)
        
        # Appliquer plusieurs couches d'attention croisée
        for _ in range(self.config.num_alignment_heads):
            # Mise à jour itérative des caractéristiques
            updated_features = []
            
            for mod in modalities:
                # Calculer l'attention par rapport aux autres modalités
                query = encoded_features[mod]
                keys = torch.cat([
                    encoded_features[m] for m in modalities if m != mod
                ], dim=0)
                
                # Attention multi-têtes
                attention_weights = F.softmax(
                    torch.matmul(query, keys.transpose(-2, -1)) / (self.config.hidden_size ** 0.5),
                    dim=-1
                )
                
                # Mise à jour pondérée
                attended = torch.matmul(attention_weights, keys)
                updated_features.append(attended)
            
            # Mettre à jour les caractéristiques encodées
            for i, mod in enumerate(modalities):
                encoded_features[mod] = self.modality_projections[mod](
                    encoded_features[mod] + updated_features[i]
                )
        
        # Combinaison finale
        combined_features = torch.stack(list(encoded_features.values())).mean(dim=0)
        return combined_features
    
    def tokenize(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize les entrées multimodales en tokens discrets.
        
        Args:
            inputs: Dictionnaire d'entrées multimodales
            
        Returns:
            Dictionnaire contenant les tokens et indices des codebooks
        """
        # Vérifier les entrées
        if not any(mod in inputs for mod in self.config.modality_dims):
            raise ValueError("Aucune entrée valide fournie. Les modalités supportées sont: "
                          f"{list(self.config.modality_dims.keys())}")
        
        # Encoder les entrées multimodales
        with torch.no_grad():
            encoded_features = self.encode_multimodal(inputs)
            
            # Aligner les modalités
            aligned_features = self.align_modalities(encoded_features)
            
            # Appliquer la quantification hiérarchique
            if hasattr(self, 'hierarchical_quantizer'):
                tokens, indices, losses = self.hierarchical_quantizer(aligned_features)
                return {
                    'tokens': tokens,
                    'indices': indices,
                    'losses': losses
                }
            
            # Tokenization par les codebooks sémantique et détaillé
            semantic_tokens, semantic_indices, semantic_loss = self.semantic_codebook(aligned_features)
            detail_tokens, detail_indices, detail_loss = self.detail_codebook(aligned_features)
            
            # Calculer la perte totale
            total_loss = semantic_loss + detail_loss
            
            return {
                'semantic_tokens': semantic_tokens,
                'semantic_indices': semantic_indices,
                'detail_tokens': detail_tokens,
                'detail_indices': detail_indices,
                'loss': total_loss,
                'aligned_features': aligned_features
            }
        
    def _compute_alignment_loss(self, encoded_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calcule une perte d'alignement pour encourager les représentations similaires entre modalités.
        Implémente un mécanisme de contrastive learning pour maximiser la similarité entre
        les représentations des mêmes échantillons dans différentes modalités.
        
        Args:
            encoded_features: Dictionnaire de caractéristiques encodées par modalité
            
        Returns:
            Perte d'alignement entre modalités
        """
        # Liste des modalités disponibles
        modalities = list(encoded_features.keys())
        num_modalities = len(modalities)
        
        # S'il n'y a qu'une seule modalité, pas d'alignement nécessaire
        if num_modalities <= 1:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Calculer la similarité cosinus entre les différentes modalités
        alignment_loss = 0.0
        num_pairs = 0
        temperature = 0.1  # Paramètre de température pour le contrastive learning
        
        for i in range(num_modalities):
            for j in range(i+1, num_modalities):
                # Récupérer les représentations des deux modalités
                feat_i = encoded_features[modalities[i]]
                feat_j = encoded_features[modalities[j]]
                
                # Vérifier que les dimensions batch correspondent
                if feat_i.size(0) != feat_j.size(0):
                    continue
                
                # Normaliser les représentations
                feat_i_norm = F.normalize(feat_i, p=2, dim=-1)
                feat_j_norm = F.normalize(feat_j, p=2, dim=-1)
                
                # Calculer les similarités pour les paires positives (même indice de batch)
                batch_size = feat_i.size(0)
                pos_sim = torch.sum(feat_i_norm * feat_j_norm, dim=-1)
                
                # Calculer toutes les similarités (positives et négatives)
                sim_matrix = torch.matmul(feat_i_norm, feat_j_norm.transpose(0, 1)) / temperature
                
                # Masque pour exclure les paires positives de la matrice de similarité
                mask = torch.eye(batch_size, device=feat_i.device)
                neg_sim = sim_matrix * (1 - mask)
                
                # InfoNCE loss: -log(exp(pos_sim) / (exp(pos_sim) + sum(exp(neg_sim))))
                pos_term = torch.exp(pos_sim / temperature)
                neg_term = torch.sum(torch.exp(neg_sim / temperature), dim=1)
                
                # Perte de contrastive learning
                nce_loss = -torch.log(pos_term / (pos_term + neg_term + 1e-8)).mean()
                alignment_loss += nce_loss
                num_pairs += 1
        
        # Moyenner sur le nombre de paires
        if num_pairs > 0:
            alignment_loss = alignment_loss / num_pairs
        
        return alignment_loss
    
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Décode les tokens en représentations continues.
        
        Args:
            tokens: Tokens à décoder
            
        Returns:
            Représentations continues décodées
        """
        return tokens  # Pour l'instant, identité - à étendre avec un décodeur
    
    def forward(self, inputs: Dict[str, Union[List[str], torch.Tensor]]) -> Dict[str, Any]:
        """
        Passage avant complet du tokenizer.
        
        Args:
            inputs: Dictionnaire d'entrées multimodales
            
        Returns:
            Résultat de la tokenization
        """
        return self.tokenize(inputs)
