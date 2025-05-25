"""
Tokenizer multimodal avancé pour l'architecture NeuroLite.

Ce module implémente un tokenizer multimodal sophistiqué qui étend les capacités
des projections multimodales existantes avec une tokenization à double codebook
et une architecture hiérarchique pour unifier la compréhension et la génération.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Any

from ..multimodal.multimodal import MultimodalProjection, CrossModalAttention
from .config import TokenizerConfig
from .quantizers import VectorQuantizer, ResidualVQ


class NeuroLiteTokenizer(nn.Module):
    """
    Tokenizer multimodal avancé pour NeuroLite.
    
    Étend les capacités de projection multimodale existantes avec une
    tokenization à double codebook (sémantique et détaillée) pour
    unifier efficacement la compréhension et la génération.
    """
    
    def __init__(self, config: TokenizerConfig):
        """
        Initialise le tokenizer multimodal NeuroLite.
        
        Args:
            config: Configuration du tokenizer
        """
        super().__init__()
        self.config = config
        
        # Utiliser la projection multimodale existante comme encodeur de base
        self.multimodal_projection = MultimodalProjection(
            output_dim=config.hidden_size,
            hidden_dim=config.hidden_size,
            dropout_rate=config.dropout_rate,
            use_cross_attention=config.use_context_modulation,
            num_attention_heads=config.num_alignment_heads,
            image_size=config.vision_encoder_config.image_size,
            patch_size=config.vision_encoder_config.patch_size,
            max_audio_length_ms=config.audio_encoder_config.max_audio_length_ms,
            max_video_frames=config.video_encoder_config.num_frames,
            max_graph_nodes=config.graph_encoder_config.max_num_nodes
        )
        
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
    
    def encode_multimodal(self, inputs: Dict[str, Union[List[str], torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Encode les entrées multimodales en représentations intermédiaires.
        
        Args:
            inputs: Dictionnaire d'entrées multimodales avec clés 'text', 'image', etc.
            
        Returns:
            Dictionnaire des représentations encodées par modalité
        """
        # Utiliser la projection multimodale existante pour traiter toutes les modalités
        # La méthode forward de MultimodalProjection peut retourner les représentations individuelles
        _, encoded_features = self.multimodal_projection(inputs, return_individual_modalities=True)
        
        # Si aucune modalité n'est présente, initialiser avec un dictionnaire vide
        if not encoded_features:
            encoded_features = {}
            
            # Créer des tenseurs vides pour chaque modalité possible si nécessaire
            device = next(self.parameters()).device
            batch_size = 1  # Par défaut
            if 'text' in inputs and inputs['text']:
                batch_size = len(inputs['text']) if isinstance(inputs['text'], list) else inputs['text'].shape[0]
            elif 'image' in inputs and inputs['image'] is not None:
                batch_size = inputs['image'].shape[0]
                
            for modality in ['text', 'image', 'audio', 'video', 'graph']:
                if modality in inputs and inputs[modality] is not None:
                    encoded_features[modality] = torch.zeros((batch_size, self.config.hidden_size), device=device)
        
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
        
        if len(modalities) == 1:
            # Une seule modalité, pas besoin d'alignement
            return list(encoded_features.values())[0]
        
        # Pour plusieurs modalités, utiliser l'attention intermodale
        aligned_features = {}
        
        # Pour chaque paire de modalités, calculer l'attention croisée
        for src_mod in modalities:
            aligned_mod_features = encoded_features[src_mod].clone()
            
            for tgt_mod in modalities:
                if src_mod != tgt_mod:
                    # Appliquer l'attention croisée entre les modalités
                    cross_attended = self.cross_modal_attention(
                        query_modality=encoded_features[src_mod],
                        key_value_modality=encoded_features[tgt_mod]
                    )
                    
                    # Mettre à jour les caractéristiques alignées
                    aligned_mod_features = aligned_mod_features + cross_attended
            
            aligned_features[src_mod] = aligned_mod_features
        
        # Combiner toutes les caractéristiques alignées (moyenne simple pour l'instant)
        combined_features = torch.stack(list(aligned_features.values())).mean(dim=0)
        
        # Normaliser et projeter
        combined_features = self.post_attention_norm(combined_features)
        combined_features = self.post_attention_projection(combined_features)
        
        return combined_features
    
    def tokenize(self, inputs: Dict[str, Union[List[str], torch.Tensor]]) -> Dict[str, Any]:
        """
        Tokenize les entrées multimodales en tokens discrets.
        
        Args:
            inputs: Dictionnaire d'entrées multimodales
            
        Returns:
            Dictionnaire contenant les tokens et indices des codebooks
        """
        # Encoder les entrées multimodales
        encoded_features = self.encode_multimodal(inputs)
        
        # Aligner les modalités
        aligned_features = self.align_modalities(encoded_features)
        
        # Tokenization par le codebook sémantique
        semantic_tokens, semantic_indices, semantic_commitment_loss = self.semantic_codebook(aligned_features)
        
        # Tokenization par le codebook détaillé
        detail_tokens, detail_indices, detail_commitment_loss = self.detail_codebook(aligned_features)
        
        # Combiner les tokens sémantiques et détaillés
        combined_tokens = torch.cat([semantic_tokens, detail_tokens], dim=-1)
        refined_tokens = self.codebook_refinement(combined_tokens)
        
        # Quantification résiduelle pour compression
        quantized, residual_indices, residual_commitment_loss = self.residual_quantizer(refined_tokens)
        
        # Projection finale
        output_tokens = self.output_projection(quantized)
        
        # Calculer la perte totale de quantification vectorielle
        vq_loss = semantic_commitment_loss + detail_commitment_loss + residual_commitment_loss
        
        # Calculer une perte d'alignement pour les fonctionnalités cross-modales
        alignment_loss = self._compute_alignment_loss(encoded_features) if len(encoded_features) > 1 else 0.0
        
        return {
            # Tokens et indices
            'semantic_tokens': semantic_tokens,
            'semantic_indices': semantic_indices,
            'detail_tokens': detail_tokens,
            'detail_indices': detail_indices,
            'refined_tokens': refined_tokens,
            'residual_indices': residual_indices,
            'output_tokens': output_tokens,
            
            # Caractéristiques extraites
            'encoded_features': encoded_features,
            'aligned_features': aligned_features,
            
            # Pertes pour l'entraînement
            'vq_loss': vq_loss,
            'semantic_commitment_loss': semantic_commitment_loss,
            'detail_commitment_loss': detail_commitment_loss,
            'residual_commitment_loss': residual_commitment_loss,
            'alignment_loss': alignment_loss,
            'commitment_loss': semantic_commitment_loss + detail_commitment_loss + residual_commitment_loss
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
