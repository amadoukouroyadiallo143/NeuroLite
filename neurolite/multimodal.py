"""
Module de projection multimodale pour NeuroLite.
Permet d'intégrer des entrées de différentes modalités (texte, image, audio)
dans l'espace de représentation commun de NeuroLite.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
from .projection import MinHashBloomProjection

class MultimodalProjection(nn.Module):
    """
    Projection multimodale pour NeuroLite.
    Convertit différentes modalités d'entrée en représentations vectorielles
    compatibles avec l'architecture NeuroLite.
    """
    
    def __init__(
        self,
        output_dim: int,
        minhash_permutations: int,
        bloom_filter_size: int,
        image_patch_size: int = 16,
        video_num_sampled_frames: int = 5, # New parameter for video
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.video_num_sampled_frames = video_num_sampled_frames
        
        # Projecteur pour le texte basé sur MinHash et Bloom
        self.text_encoder = MinHashBloomProjection(
            output_dim=output_dim,
            minhash_permutations=minhash_permutations,
            bloom_filter_size=bloom_filter_size,
            dropout_rate=dropout_rate
        )
        
        # Encodeur d'images allégé (inspiré des ViT minimalistes)
        self.image_encoder = nn.Sequential(
            # Convertir l'image en patches et les projeter
            nn.Conv2d(3, 16, kernel_size=image_patch_size, stride=image_patch_size),
            nn.LayerNorm([16, 14, 14]),  # Pour images 224x224
            nn.GELU(),
            nn.Flatten(1),  # [B, 16*14*14]
            nn.Linear(16*14*14, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Encodeur audio simplifié (basé sur des features spectrales)
        self.audio_encoder = nn.Sequential(
            # Expectation: Mel-spectrogramme prétraité [B, 1, T, F]
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([16, 64, 40]),  # Pour spectrogrammes standard
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.LayerNorm([32, 15, 9]),
            nn.GELU(),
            nn.Flatten(1),
            nn.Linear(32*15*9, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout_rate)
        )

        # Encodeur pour les frames vidéo individuelles
        # Re-utilise une architecture similaire à image_encoder.
        # Si les frames vidéo ont des dimensions standard différentes (ex: 112x112 au lieu de 224x224),
        # les dimensions de LayerNorm et Linear devront être ajustées.
        # Pour cet exemple, on suppose que les frames vidéo sont traitées comme des images.
        # Si image_patch_size est 16:
        # Pour 224x224 -> 14x14 patches. 16*14*14 = 3136
        # Pour 112x112 -> 7x7 patches.   16*7*7 = 784
        # Let's assume video frames might be smaller, e.g., 112x112, leading to 7x7 patches.
        # We will define a separate video_frame_processor for clarity,
        # but it could share layers or be identical to image_encoder if frames are same size.
        
        # Assuming video frames are, for example, 112x112, patch_size=16 -> 7x7 patches
        # Conv2d(3, 16, kernel_size=16, stride=16) -> [B, 16, 7, 7]
        # LayerNorm([16, 7, 7])
        # Flatten(1) -> [B, 16*7*7 = 784]
        # Linear(784, output_dim)
        # This structure is illustrative. For simplicity, we can make it identical to image_encoder
        # and assume input frames will be resized to what image_encoder expects (e.g. 224x224).
        # Or, make it configurable. For now, let's use a distinct path but similar structure.
        # For this task, we'll keep it simple and assume video frames can be processed by a similar architecture
        # to the image_encoder. For robustness, let's use the same image_encoder for frames.
        self.video_frame_processor = self.image_encoder 
        # Note: If video frames have significantly different characteristics or typical resolutions than standalone images,
        # a dedicated architecture for video_frame_processor would be better. Reusing image_encoder implies
        # video frames are treated like individual images of the same expected input size.

        # Frame Aggregation: Using mean pooling, so no specific nn.Module needed here,
        # it will be done in the forward pass. If using Linear/RNN, it would be defined here.
        
        # Fusion adaptative des modalités (maintenant 4 modalités)
        self.fusion_weights = nn.Parameter(torch.ones(4) / 4.0)  # Texte, Image, Audio, Vidéo
        self.fusion_gate = nn.Sequential(
            nn.Linear(output_dim * 4, 4), # Input_dim * 4
            nn.Softmax(dim=-1)
        )
    
    def forward(self, 
                inputs: Dict[str, Union[List[str], torch.Tensor, None]], 
                return_individual_modalities: bool = False
               ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Traite des entrées multimodales et produit une représentation commune.
        
        Args:
            inputs: Dictionnaire avec les clés 'text', 'image', 'audio', 'video'
                    contenant les données pour chaque modalité.
            return_individual_modalities: Si True, retourne aussi un dictionnaire
                                           des représentations de chaque modalité.
                   
        Returns:
            - Si return_individual_modalities est False: Tensor de représentations fusionnées [batch_size, output_dim]
            - Si return_individual_modalities est True: Tuple (fused_representation, individual_representations_dict)
        """
        batch_size = self._get_batch_size(inputs)
        device = self._get_device(inputs)
        
        individual_representations: Dict[str, torch.Tensor] = {}
        
        
        # Initialiser les représentations par modalité
        # Utiliser des tenseurs nuls par défaut et les remplacer si la modalité est présente
        text_repr = torch.zeros((batch_size, self.output_dim), device=device)
        image_repr = torch.zeros((batch_size, self.output_dim), device=device)
        audio_repr = torch.zeros((batch_size, self.output_dim), device=device)
        video_repr = torch.zeros((batch_size, self.output_dim), device=device)
        
        # Encoder chaque modalité si présente
        if "text" in inputs and inputs["text"] is not None and len(inputs["text"]) > 0:
            text_repr = self.text_encoder(inputs["text"])
            if torch.any(text_repr != 0): individual_representations["text"] = text_repr
            
        if "image" in inputs and inputs["image"] is not None:
            image_repr = self.image_encoder(inputs["image"])
            if torch.any(image_repr != 0): individual_representations["image"] = image_repr
            
        if "audio" in inputs and inputs["audio"] is not None:
            audio_repr = self.audio_encoder(inputs["audio"])
            if torch.any(audio_repr != 0): individual_representations["audio"] = audio_repr

        if "video" in inputs and inputs["video"] is not None:
            video_input = inputs["video"] 
            num_total_frames = video_input.size(1)
            sampled_frames = None # Initialize to avoid potential reference before assignment

            if num_total_frames == 0:
                 pass 
            elif num_total_frames <= self.video_num_sampled_frames:
                sampled_frames = video_input
            else:
                indices = torch.linspace(0, num_total_frames - 1, self.video_num_sampled_frames, device=device).long()
                sampled_frames = video_input.index_select(1, indices)

            if sampled_frames is not None and sampled_frames.numel() > 0 :
                b, n_sampled, c, h, w = sampled_frames.shape
                frames_reshaped = sampled_frames.reshape(b * n_sampled, c, h, w)
                frame_features = self.video_frame_processor(frames_reshaped)
                frame_features_orig_shape = frame_features.view(b, n_sampled, self.output_dim)
                video_repr = frame_features_orig_shape.mean(dim=1)
                if torch.any(video_repr != 0): individual_representations["video"] = video_repr
        
        # Fusion adaptative par gate mechanism
        # Utiliser les représentations calculées (qui peuvent être zéro si la modalité est absente)
        combined_input_for_gate = torch.cat([text_repr, image_repr, audio_repr, video_repr], dim=-1)
        fusion_weights = self.fusion_gate(combined_input_for_gate) 
        
        fused_repr = (
            fusion_weights[..., 0:1] * text_repr +
            fusion_weights[..., 1:2] * image_repr +
            fusion_weights[..., 2:3] * audio_repr +
            fusion_weights[..., 3:4] * video_repr
        )
        
        if return_individual_modalities:
            return fused_repr, individual_representations
        else:
            return fused_repr
    
    def _get_batch_size(self, inputs: Dict[str, Union[List[str], torch.Tensor, None]]) -> int:
        """Détermine la taille du batch à partir des entrées"""
        if "text" in inputs and inputs["text"] is not None and len(inputs["text"]) > 0:
            return len(inputs["text"])
        elif "image" in inputs and inputs["image"] is not None:
            return inputs["image"].size(0)
        elif "audio" in inputs and inputs["audio"] is not None:
            return inputs["audio"].size(0)
        elif "video" in inputs and inputs["video"] is not None: # Added video check
            return inputs["video"].size(0)
        else:
            # Attempt to find a valid input to infer batch size if primary ones are missing
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor) and inputs[key] is not None:
                    return inputs[key].size(0)
                elif isinstance(inputs[key], list) and len(inputs[key]) > 0:
                    return len(inputs[key])
            return 1  # Par défaut if no valid input found
            
    def _get_device(self, inputs: Dict[str, Union[List[str], torch.Tensor, None]]) -> torch.device:
        """Détermine le device des tenseurs d'entrée"""
        if "image" in inputs and inputs["image"] is not None:
            return inputs["image"].device
        elif "audio" in inputs and inputs["audio"] is not None:
            return inputs["audio"].device
        elif "video" in inputs and inputs["video"] is not None: # Added video check
            return inputs["video"].device
        else:
            # Attempt to find a valid tensor input to infer device
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor) and inputs[key] is not None:
                    return inputs[key].device
            return torch.device("cpu")  # Par défaut


class CrossModalAttention(nn.Module):
    """
    Module d'attention cross-modale pour fusionner des informations
    entre différentes modalités.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        # Projections pour l'attention
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(
        self,
        query_modality: torch.Tensor,
        key_value_modality: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Applique l'attention cross-modale.
        
        Args:
            query_modality: Tensor de la modalité de requête [batch, seq_len_q, hidden_size]
            key_value_modality: Tensor de la modalité clé/valeur [batch, seq_len_kv, hidden_size]
            attention_mask: Masque d'attention optionnel [batch, seq_len_q, seq_len_kv]
            
        Returns:
            Tensor fusionné [batch, seq_len_q, hidden_size]
        """
        residual = query_modality
        
        batch_size, seq_len_q, _ = query_modality.shape
        _, seq_len_kv, _ = key_value_modality.shape
        
        # Projections
        q = self.q_proj(query_modality).view(
            batch_size, seq_len_q, self.num_heads, self.head_size
        ).transpose(1, 2)  # [batch, heads, seq_len_q, head_size]
        
        k = self.k_proj(key_value_modality).view(
            batch_size, seq_len_kv, self.num_heads, self.head_size
        ).transpose(1, 2)  # [batch, heads, seq_len_kv, head_size]
        
        v = self.v_proj(key_value_modality).view(
            batch_size, seq_len_kv, self.num_heads, self.head_size
        ).transpose(1, 2)  # [batch, heads, seq_len_kv, head_size]
        
        # Calcul de l'attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_size ** 0.5)
        
        if attention_mask is not None:
            # Appliquer le masque (ajouter -inf où le masque est 0)
            scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, float("-inf"))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Appliquer l'attention aux valeurs
        context = torch.matmul(attn_weights, v)  # [batch, heads, seq_len_q, head_size]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.hidden_size
        )  # [batch, seq_len_q, hidden_size]
        
        # Projection de sortie et connexion résiduelle
        output = self.output_proj(context)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)
        
        return output
