"""
Décodeur spécialisé pour la génération de texte.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union, Tuple, Any

from neurolite.core.ssm import SSMLayer
from neurolite.Configs.config import MMTextDecoderConfig
from .base_decoder import BaseDecoder

class TextDecoder(BaseDecoder):
    """
    Décodeur pour la génération de texte à partir d'une représentation latente.
    """
    
    def __init__(self, config: MMTextDecoderConfig):
        """
        Initialise le décodeur de texte.
        
        Args:
            config: Objet de configuration MMTextDecoderConfig.
        """
        super().__init__(config)
        self.config = config
        
        # Projection de l'entrée latente
        self.input_projection = nn.Linear(config.input_dim, config.embedding_dim)
        
        # Embedding de tokens
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        # Encodage positionnel
        if config.use_positional_encoding:
            self.positional_encoding = nn.Parameter(
                torch.zeros(1, config.max_seq_len, config.embedding_dim)
            )
            self.register_buffer(
                "position_ids", torch.arange(config.max_seq_len).expand((1, -1))
            )
        
        # Couches décodeur
        if config.use_ssm_layers:
            ssm_layers_list = [
                SSMLayer(
                    dim=config.embedding_dim, 
                    d_state=config.ssm_d_state, 
                    d_conv=config.ssm_d_conv, 
                    expand_factor=config.ssm_expand_factor
                ) for _ in range(config.num_layers)
            ]
            self.decoder_layers = nn.Sequential(*ssm_layers_list)
        else:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim, 
                dropout=config.dropout_rate,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
            self.decoder_layers = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=config.num_layers
            )
        
        # Projection de sortie pour la génération de tokens
        self.output_projection = nn.Linear(config.embedding_dim, config.vocab_size)
        
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
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Décode une représentation latente en une séquence de logits.
        Si des cibles sont fournies (mode entraînement), calcule aussi la perte.

        Args:
            latent: Représentation latente [batch_size, seq_length, input_dim].
            targets: Séquence de tokens cibles (input_ids) [batch_size, seq_length].

        Returns:
            Un dictionnaire contenant :
            - 'logits': Logits pour chaque token [batch_size, seq_length, vocab_size].
            - 'loss': Perte de cross-entropy (si `targets` est fourni).
        """
        if targets is None:
            # La logique de génération (inférence) est gérée par la méthode `generate`.
            # Cette méthode `forward` est principalement pour l'entraînement.
            raise ValueError("Le TextDecoder en mode forward (entraînement) requiert des `targets` (input_ids).")

        # --- Préparation des entrées et des labels ---
        
        # Remplacer l'index d'ignorance (-100) par le pad_token_id avant l'embedding.
        # L'embedding ne peut pas gérer d'indices négatifs.
        input_ids = targets.clone()
        input_ids[input_ids == -100] = self.config.pad_token_id
        
        # Pour la prédiction du token suivant, le modèle voit les tokens 0..n-1
        # et doit prédire les tokens 1..n.
        decoder_input_ids = input_ids[:, :-1]
        lm_labels = targets[:, 1:].contiguous() # Utiliser les cibles originales pour la perte
        
        batch_size, seq_length = decoder_input_ids.shape
        device = decoder_input_ids.device
        
        # --- Création des masques ---

        # 1. Masque causal pour empêcher l'attention sur les tokens futurs.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_length, device=device)

        # 2. Masque de padding pour ignorer les tokens de padding.
        padding_mask = (decoder_input_ids == self.config.pad_token_id)
        
        # --- Préparation des représentations ---

        # 1. Projeter le latent pour qu'il serve de "mémoire" au décodeur.
        # La forme devient [batch_size, seq_len, embedding_dim]
        memory = self.input_projection(latent)
        
        # 2. Obtenir les embeddings des tokens d'entrée.
        token_embeds = self.token_embedding(decoder_input_ids)
        
        # 3. Ajouter l'encodage positionnel.
        if self.config.use_positional_encoding:
            token_embeds = token_embeds + self.positional_encoding[:, :seq_length, :]

        # --- Décodage ---
        
        if self.config.use_ssm_layers:
            # TODO: Implémenter la logique pour les couches SSM avec une mémoire cross-attention
            raise NotImplementedError("La logique du décodeur SSM avec mémoire n'est pas encore implémentée.")
        else:
            # Le TransformerDecoder de PyTorch attend `memory` avec la forme [batch_size, src_seq_len, dim].
            # Ici, notre `src_seq_len` est 1.
            output = self.decoder_layers(
                tgt=token_embeds,
                memory=memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=None # Le latent n'a pas de padding
            )

        # --- Calcul de la sortie et de la perte ---
        
        # Projection finale vers l'espace du vocabulaire
        logits = self.output_projection(output)
        
        loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            lm_labels.view(-1),
            ignore_index=-100 # Ignorer les prompts et le padding dans le calcul de la perte
        )
        
        return {
            'logits': logits, # 'output' a été renommé en 'logits' pour plus de clarté
            'loss': loss
        }
    
    @torch.no_grad()
    def generate(
        self,
        latent: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_beams: int = 1,
        do_sample: bool = True,
        **kwargs # Accepter les arguments supplémentaires pour la flexibilité
    ) -> torch.Tensor:
        """
        Génère du texte à partir d'une représentation latente.
        """
        batch_size = latent.shape[0]
        device = latent.device
        
        # Projeter la séquence latente pour qu'elle serve de mémoire
        memory = self.input_projection(latent)
        
        input_ids = torch.full(
            (batch_size, 1), self.config.bos_token_id, dtype=torch.long, device=device
        )

        if num_beams > 1:
            # La recherche par faisceau est plus complexe
            # Pour simplifier, nous utilisons la génération gloutonne/échantillonnage ici.
            # Une implémentation complète de beam search serait nécessaire pour la production.
            pass

        for _ in range(max_length - 1):
            seq_len = input_ids.shape[1]
            
            # Préparation des masques pour le forward du transformer
            token_embeds = self.token_embedding(input_ids)
            if self.config.use_positional_encoding:
                token_embeds += self.positional_encoding[:, :seq_len, :]

            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
            
            # Forward pass
            if self.config.use_ssm_layers:
                 output = self.decoder_layers(token_embeds + memory)
            else:
                # Pour la génération, memory est constant à chaque étape
                output = self.decoder_layers(
                    tgt=token_embeds,
                    memory=memory,
                    tgt_mask=causal_mask
                )

            # Obtenir les logits du dernier token seulement
            next_token_logits = self.output_projection(output[:, -1, :])

            # Appliquer la température
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Top-k/Top-p sampling
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')

            # Échantillonner le prochain token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Ajouter le token prédit à la séquence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Condition d'arrêt
            if next_token.item() == self.config.eos_token_id:
                break
                
        return input_ids

    def _update_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if hasattr(module.weight, 'data'):
                module.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)
