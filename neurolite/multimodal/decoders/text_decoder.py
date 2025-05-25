"""
Décodeur spécialisé pour la génération de texte.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union, Tuple, Any

class TextDecoder(nn.Module):
    """
    Décodeur pour la génération de texte à partir d'une représentation latente.
    """
    
    def __init__(
        self,
        input_dim: int,
        vocab_size: int = 50000,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        max_seq_length: int = 2048,
        use_positional_encoding: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        bos_token_id: int = 1
    ):
        """
        Initialise le décodeur de texte.
        
        Args:
            input_dim: Dimension d'entrée
            vocab_size: Taille du vocabulaire
            embedding_dim: Dimension des embeddings
            hidden_dim: Dimension cachée
            num_layers: Nombre de couches transformer
            num_heads: Nombre de têtes d'attention
            dropout_rate: Taux de dropout
            max_seq_length: Longueur maximale de séquence
            use_positional_encoding: Si True, utilise l'encodage positionnel
            pad_token_id: ID du token de padding
            eos_token_id: ID du token de fin de séquence
            bos_token_id: ID du token de début de séquence
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        
        # Projection de l'entrée latente
        self.input_projection = nn.Linear(input_dim, embedding_dim)
        
        # Embedding de tokens
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Encodage positionnel
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.positional_encoding = nn.Parameter(
                torch.zeros(1, max_seq_length, embedding_dim)
            )
            nn.init.trunc_normal_(self.positional_encoding, std=0.02)
        
        # Couches décodeur Transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )
        
        # Projection de sortie pour la génération de tokens
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
        # Liaison des poids d'embedding et de projection (weight tying)
        self.output_projection.weight = self.token_embedding.weight
        
        # Initialisation des poids
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialise les poids du modèle."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        latent: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Décode une représentation latente en séquence de tokens.
        
        Args:
            latent: Représentation latente [batch_size, input_dim]
            input_ids: IDs des tokens d'entrée [batch_size, seq_length]
            attention_mask: Masque d'attention [batch_size, seq_length]
            
        Returns:
            Logits pour chaque position [batch_size, seq_length, vocab_size]
        """
        batch_size, seq_length = input_ids.shape
        
        # Projection de l'entrée latente
        memory = self.input_projection(latent).unsqueeze(1)  # [batch_size, 1, embedding_dim]
        
        # Embeddings des tokens
        token_embeds = self.token_embedding(input_ids)  # [batch_size, seq_length, embedding_dim]
        
        # Ajouter l'encodage positionnel
        if self.use_positional_encoding:
            token_embeds = token_embeds + self.positional_encoding[:, :seq_length, :]
        
        # Créer les masques d'attention
        if attention_mask is not None:
            # Masque de padding pour l'entrée
            padding_mask = (attention_mask == 0)
            
            # Masque causal pour l'auto-attention
            causal_mask = torch.triu(
                torch.ones(seq_length, seq_length, device=input_ids.device),
                diagonal=1
            ).bool()
        else:
            padding_mask = None
            causal_mask = torch.triu(
                torch.ones(seq_length, seq_length, device=input_ids.device),
                diagonal=1
            ).bool()
        
        # Décodage Transformer
        if padding_mask is not None:
            output = self.transformer_decoder(
                token_embeds,
                memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=padding_mask
            )
        else:
            output = self.transformer_decoder(
                token_embeds,
                memory,
                tgt_mask=causal_mask
            )
        
        # Projection finale
        logits = self.output_projection(output)
        
        return logits
    
    def generate(
        self,
        latent: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_return_sequences: int = 1,
        num_beams: Optional[int] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Génère du texte à partir d'une représentation latente.
        
        Args:
            latent: Représentation latente [batch_size, input_dim]
            max_length: Longueur maximale de la séquence générée
            temperature: Température pour l'échantillonnage
            top_k: Nombre de tokens les plus probables à considérer
            top_p: Seuil de probabilité cumulée (nucleus sampling)
            num_return_sequences: Nombre de séquences à retourner
            num_beams: Nombre de faisceaux pour la recherche par faisceau
            do_sample: Si True, échantillonne selon les probabilités
            
        Returns:
            Séquences générées [batch_size * num_return_sequences, seq_length]
        """
        batch_size = latent.shape[0]
        device = latent.device
        
        # Projection de l'entrée latente
        memory = self.input_projection(latent).unsqueeze(1)  # [batch_size, 1, embedding_dim]
        
        # Initialiser avec le token de début de séquence
        input_ids = torch.full(
            (batch_size, 1),
            self.bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Pour la recherche par faisceau
        if num_beams is not None and num_beams > 1:
            return self._generate_beam_search(
                input_ids, memory, max_length, num_beams, num_return_sequences
            )
        
        # Génération auto-régressive
        for _ in range(max_length):
            # Obtenir les embeddings
            token_embeds = self.token_embedding(input_ids)  # [batch_size, curr_length, embedding_dim]
            
            # Ajouter l'encodage positionnel
            if self.use_positional_encoding:
                curr_length = token_embeds.size(1)
                token_embeds = token_embeds + self.positional_encoding[:, :curr_length, :]
            
            # Masque causal pour l'auto-attention
            causal_mask = torch.triu(
                torch.ones(curr_length, curr_length, device=device),
                diagonal=1
            ).bool()
            
            # Décodage Transformer
            output = self.transformer_decoder(
                token_embeds,
                memory,
                tgt_mask=causal_mask
            )
            
            # Prédire le prochain token
            next_token_logits = self.output_projection(output[:, -1, :])
            
            # Appliquer la température
            next_token_logits = next_token_logits / temperature
            
            # Échantillonnage du prochain token
            if do_sample:
                # Top-k sampling
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Top-p (nucleus) sampling
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Supprimer les tokens avec une probabilité cumulative > top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Conserver le premier token au-dessus du seuil
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Échantillonner à partir des logits
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # Argmax
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Ajouter le token à la séquence
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            # Vérifier si tous les exemples ont généré un token EOS
            eos_generated = (next_tokens.squeeze(-1) == self.eos_token_id)
            if eos_generated.all():
                break
        
        return input_ids
    
    def _generate_beam_search(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        max_length: int,
        num_beams: int,
        num_return_sequences: int
    ) -> torch.Tensor:
        """
        Génère des séquences avec la recherche par faisceau.
        Implémentation simplifiée de la recherche par faisceau.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Répliquer l'entrée pour chaque faisceau
        expanded_input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        expanded_memory = memory.repeat_interleave(num_beams, dim=0)
        
        # Scores des faisceaux
        beam_scores = torch.zeros((batch_size, num_beams), device=device)
        
        # Liste pour stocker les séquences terminées
        done_sequences = [[] for _ in range(batch_size)]
        done_scores = [[] for _ in range(batch_size)]
        
        # Génération auto-régressive
        for step in range(max_length):
            # Obtenir les embeddings
            token_embeds = self.token_embedding(expanded_input_ids)
            
            # Ajouter l'encodage positionnel
            if self.use_positional_encoding:
                curr_length = token_embeds.size(1)
                token_embeds = token_embeds + self.positional_encoding[:, :curr_length, :]
            
            # Masque causal pour l'auto-attention
            causal_mask = torch.triu(
                torch.ones(curr_length, curr_length, device=device),
                diagonal=1
            ).bool()
            
            # Décodage Transformer
            output = self.transformer_decoder(
                token_embeds,
                expanded_memory,
                tgt_mask=causal_mask
            )
            
            # Prédire le prochain token
            next_token_logits = self.output_projection(output[:, -1, :])
            vocab_size = next_token_logits.shape[-1]
            
            # Calculer les probabilités
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # [batch_size*num_beams, vocab_size]
            
            # Reshape pour le traitement par lot
            next_token_scores = next_token_scores.view(batch_size, num_beams, -1)  # [batch_size, num_beams, vocab_size]
            
            # Ajouter les scores du faisceau courant
            next_scores = beam_scores.unsqueeze(-1) + next_token_scores  # [batch_size, num_beams, vocab_size]
            
            # Flatten pour faciliter le top-k
            next_scores = next_scores.view(batch_size, -1)  # [batch_size, num_beams*vocab_size]
            
            # Obtenir les top-k tokens
            topk_scores, topk_indices = torch.topk(next_scores, num_beams, dim=1)
            
            # Calculer quels faisceaux et tokens ont été sélectionnés
            beam_indices = topk_indices // vocab_size
            token_indices = topk_indices % vocab_size
            
            # Mettre à jour les scores des faisceaux
            beam_scores = topk_scores
            
            # Générer les nouveaux input_ids
            new_input_ids = []
            for batch_idx in range(batch_size):
                # Récupérer les indices des faisceaux sélectionnés
                batch_beam_indices = beam_indices[batch_idx]
                
                for beam_idx, token_idx in zip(batch_beam_indices, token_indices[batch_idx]):
                    # Récupérer la séquence courante
                    beam_idx = beam_idx.item()
                    token_idx = token_idx.item()
                    
                    # Indice dans l'expanded_input_ids
                    idx = batch_idx * num_beams + beam_idx
                    
                    # Séquence courante
                    current_seq = expanded_input_ids[idx].clone()
                    
                    # Vérifier si le token est EOS
                    if token_idx == self.eos_token_id:
                        done_sequences[batch_idx].append(torch.cat([current_seq, torch.tensor([token_idx], device=device)]))
                        done_scores[batch_idx].append(beam_scores[batch_idx, beam_idx])
                    else:
                        # Ajouter le token à la séquence
                        new_seq = torch.cat([current_seq, torch.tensor([token_idx], device=device)])
                        new_input_ids.append(new_seq)
            
            # Si toutes les séquences sont terminées, sortir
            if all(len(done) >= num_return_sequences for done in done_sequences):
                break
            
            # Mettre à jour les input_ids
            if new_input_ids:
                expanded_input_ids = torch.stack(new_input_ids)
            else:
                break
        
        # Sélectionner les meilleures séquences
        result = []
        for batch_idx in range(batch_size):
            sequences = done_sequences[batch_idx]
            scores = done_scores[batch_idx]
            
            # Si aucune séquence n'est terminée, prendre les séquences actuelles
            if not sequences:
                for i in range(num_beams):
                    idx = batch_idx * num_beams + i
                    sequences.append(expanded_input_ids[idx])
                    scores.append(beam_scores[batch_idx, i])
            
            # Trier par score
            sorted_seqs_scores = sorted(zip(sequences, scores), key=lambda x: -x[1])
            top_seqs = [seq for seq, _ in sorted_seqs_scores[:num_return_sequences]]
            
            # Padding à la même longueur
            max_len = max(seq.size(0) for seq in top_seqs)
            padded_seqs = []
            for seq in top_seqs:
                padded_seq = F.pad(
                    seq, (0, max_len - seq.size(0)), value=self.pad_token_id
                )
                padded_seqs.append(padded_seq)
            
            # Concaténer les séquences
            batch_result = torch.stack(padded_seqs)
            result.append(batch_result)
        
        # Concaténer les résultats de tous les lots
        return torch.cat(result, dim=0)
