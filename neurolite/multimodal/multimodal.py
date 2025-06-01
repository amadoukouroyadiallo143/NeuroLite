"""
Module de projection multimodale avancé pour NeuroLite.
Permet d'intégrer des entrées de différentes modalités (texte, image, audio, vidéo, graphe)
dans l'espace de représentation commun de NeuroLite, utilisant des encodeurs et
décodeurs spécialisés pour chaque modalité.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Any

# Importation des encodeurs spécialisés
from .encoders.text_encoder import TextEncoder
from .encoders.image_encoder import ImageEncoder
from .encoders.audio_encoder import AudioEncoder
from .encoders.video_encoder import VideoEncoder
from .encoders.graph_encoder import GraphEncoder

# Importation des décodeurs spécialisés
from .decoders.text_decoder import TextDecoder
from .decoders.image_decoder import ImageDecoder
from .decoders.audio_decoder import AudioDecoder
from .decoders.video_decoder import VideoDecoder
from .decoders.graph_decoder import GraphDecoder


class MultimodalProjection(nn.Module):
    """
    Projection multimodale avancée pour NeuroLite.
    Convertit différentes modalités d'entrée en représentations vectorielles
    compatibles avec l'architecture NeuroLite.
    """
    
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 768,
        dropout_rate: float = 0.1,
        use_cross_attention: bool = True,
        num_attention_heads: int = 8,
        image_size: int = 224,
        patch_size: int = 16,
        max_audio_length_ms: int = 30000,
        max_video_frames: int = 32,
        max_graph_nodes: int = 32
    ):
        """
        Initialise le module de projection multimodale.
        
        Args:
            output_dim: Dimension de sortie commune
            hidden_dim: Dimension cachée interne
            dropout_rate: Taux de dropout
            use_cross_attention: Si True, utilise l'attention croisée pour la fusion
            num_attention_heads: Nombre de têtes d'attention
            image_size: Taille des images d'entrée (supposées carrées)
            patch_size: Taille des patches pour les images et vidéos
            max_audio_length_ms: Durée maximale de l'audio en millisecondes
            max_video_frames: Nombre maximal de trames vidéo
            max_graph_nodes: Nombre maximal de nœuds dans un graphe
        """
        super().__init__()
        
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_cross_attention = use_cross_attention
        
        # Encodeurs spécialisés
        self.text_encoder = TextEncoder(
            output_dim=hidden_dim,
            embedding_dim=256,
            hidden_dim=512,
            num_layers=3,
            num_heads=8,
            dropout_rate=dropout_rate,
            pooling_method="mean"
        )
        
        self.image_encoder = ImageEncoder(
            output_dim=hidden_dim,
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=hidden_dim,
            depth=12,
            num_heads=num_attention_heads,
            dropout_rate=dropout_rate,
            pooling_method="cls"
        )
        
        self.audio_encoder = AudioEncoder(
            output_dim=hidden_dim,
            sample_rate=16000,
            feature_type="mel_spectrogram",
            embed_dim=hidden_dim,
            depth=6,
            num_heads=num_attention_heads,
            dropout_rate=dropout_rate,
            max_audio_length_ms=max_audio_length_ms,
            pooling_method="mean"
        )
        
        self.video_encoder = VideoEncoder(
            output_dim=hidden_dim,
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=hidden_dim,
            image_encoder_depth=12,
            temporal_depth=4,
            num_heads=num_attention_heads,
            dropout_rate=dropout_rate,
            max_frames=max_video_frames,
            temporal_pooling_method="mean"
        )
        
        self.graph_encoder = GraphEncoder(
            output_dim=hidden_dim,
            node_feature_dim=64,
            hidden_dim=hidden_dim,
            num_layers=3,
            num_heads=num_attention_heads,
            dropout_rate=dropout_rate,
            readout_method="mean"
        )
        
        # Projections finales vers l'espace commun
        self.text_projection = nn.Linear(hidden_dim, output_dim)
        self.image_projection = nn.Linear(hidden_dim, output_dim)
        self.audio_projection = nn.Linear(hidden_dim, output_dim)
        self.video_projection = nn.Linear(hidden_dim, output_dim)
        self.graph_projection = nn.Linear(hidden_dim, output_dim)
        
        # Module d'attention croisée pour la fusion des modalités
        if use_cross_attention:
            self.cross_attention = CrossModalAttention(
                hidden_size=hidden_dim,
                num_heads=num_attention_heads,
                dropout_rate=dropout_rate
            )
        
        # Fusion adaptative des modalités
        self.fusion_weights = nn.Parameter(torch.ones(5) / 5.0)  # Texte, Image, Audio, Vidéo, Graphe
        self.fusion_gate = nn.Sequential(
            nn.Linear(output_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 5),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, 
                inputs: Dict[str, Union[List[str], torch.Tensor, None]], 
                return_individual_modalities: bool = False
               ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Traite des entrées multimodales et produit une représentation commune.
        
        Args:
            inputs: Dictionnaire avec les clés 'text', 'image', 'audio', 'video', 'graph'
                   contenant les données pour chaque modalité.
            return_individual_modalities: Si True, retourne aussi un dictionnaire
                                          des représentations de chaque modalité.
                   
        Returns:
            - Si return_individual_modalities est False: Tensor de représentations fusionnées [batch_size, output_dim]
            - Si return_individual_modalities est True: Tuple (fused_representation, individual_representations_dict)
        """
        batch_size = self._get_batch_size(inputs)
        device = self._get_device(inputs)
        
        batch_size = self._get_batch_size(inputs)
        device = self._get_device(inputs)

        # Step 1: Encode all modalities to hidden_dim (raw representations)
        raw_modality_reprs: Dict[str, Optional[torch.Tensor]] = {
            "text": None, "image": None, "audio": None, "video": None, "graph": None
        }

        if "text" in inputs and inputs["text"] is not None and len(inputs["text"]) > 0:
            raw_modality_reprs["text"] = self.text_encoder(inputs["text"])
        if "image" in inputs and inputs["image"] is not None:
            raw_modality_reprs["image"] = self.image_encoder(inputs["image"])
        if "audio" in inputs and inputs["audio"] is not None:
            raw_modality_reprs["audio"] = self.audio_encoder(inputs["audio"])
        if "video" in inputs and inputs["video"] is not None:
            raw_modality_reprs["video"] = self.video_encoder(inputs["video"])
        if "graph" in inputs and inputs["graph"] is not None:
            node_features = inputs["graph"].get("node_features")
            adjacency_matrix = inputs["graph"].get("adjacency_matrix")
            node_mask = inputs["graph"].get("node_mask")
            if node_features is not None and adjacency_matrix is not None:
                raw_modality_reprs["graph"] = self.graph_encoder(node_features, adjacency_matrix, node_mask)

        # Step 2: Apply internal cross-attention if enabled and multiple modalities exist
        # This cross-attention operates on hidden_dim representations.
        processed_modality_reprs = raw_modality_reprs.copy()

        active_raw_reprs = {m: r for m, r in raw_modality_reprs.items() if r is not None and torch.any(r != 0)}

        if self.use_cross_attention and len(active_raw_reprs) > 1:
            # Create a list of tensors for cross-attention input
            # The order should be consistent (e.g., text, image, audio, video, graph)
            ordered_active_keys = [m for m in ["text", "image", "audio", "video", "graph"] if m in active_raw_reprs]
            
            # Tensors for cross-attention, all shaped [batch_size, 1, hidden_dim]
            # CrossModalAttention expects [batch, seq_len, hidden_size]
            # Here, each modality is a single "token" in a sequence of modalities.
            cross_attn_input_list = [active_raw_reprs[key].unsqueeze(1) for key in ordered_active_keys]

            # Stack them to form a sequence of modalities: [batch_size, num_active_modalities, hidden_dim]
            if len(cross_attn_input_list) > 0:
                stacked_reprs = torch.cat(cross_attn_input_list, dim=1)

                # Enhance each modality by attending to all others
                enhanced_repr_list = []
                for i in range(len(ordered_active_keys)):
                    query_modality_tensor = stacked_reprs[:, i:i+1, :] # [batch, 1, hidden_dim]
                    # Key/value is the full stack of modalities
                    # An attention mask can prevent attending to oneself, though CrossModalAttention's residual connection handles this.
                    # For simplicity, allow self-attention within this context.
                    # Or, create a mask:
                    # mask = torch.ones(batch_size, 1, len(ordered_active_keys), device=device)
                    # mask[:, :, i] = 0 # Query attends to all others, not itself.
                    
                    # Using self.cross_attention (which takes query, key_value)
                    # Here, query is one modality, key_value is the set of all modalities
                    enhanced_modality_tensor = self.cross_attention(query_modality_tensor, stacked_reprs) #, attention_mask=mask)
                    enhanced_repr_list.append(enhanced_modality_tensor.squeeze(1)) # Back to [batch, hidden_dim]
                
                # Update the processed_modality_reprs with enhanced versions
                for idx, key in enumerate(ordered_active_keys):
                    processed_modality_reprs[key] = enhanced_repr_list[idx]

        # Step 3: Project all (raw or cross-attended) representations to output_dim
        # These will be returned as individual_representations if requested
        final_individual_representations: Dict[str, torch.Tensor] = {}
        # Initialize with zero tensors in output_dim for fusion gate
        text_proj = torch.zeros((batch_size, self.output_dim), device=device)
        image_proj = torch.zeros((batch_size, self.output_dim), device=device)
        audio_proj = torch.zeros((batch_size, self.output_dim), device=device)
        video_proj = torch.zeros((batch_size, self.output_dim), device=device)
        graph_proj = torch.zeros((batch_size, self.output_dim), device=device)

        if processed_modality_reprs["text"] is not None and torch.any(processed_modality_reprs["text"] != 0):
            text_proj = self.text_projection(processed_modality_reprs["text"])
            final_individual_representations["text"] = text_proj
        if processed_modality_reprs["image"] is not None and torch.any(processed_modality_reprs["image"] != 0):
            image_proj = self.image_projection(processed_modality_reprs["image"])
            final_individual_representations["image"] = image_proj
        if processed_modality_reprs["audio"] is not None and torch.any(processed_modality_reprs["audio"] != 0):
            audio_proj = self.audio_projection(processed_modality_reprs["audio"])
            final_individual_representations["audio"] = audio_proj
        if processed_modality_reprs["video"] is not None and torch.any(processed_modality_reprs["video"] != 0):
            video_proj = self.video_projection(processed_modality_reprs["video"])
            final_individual_representations["video"] = video_proj
        if processed_modality_reprs["graph"] is not None and torch.any(processed_modality_reprs["graph"] != 0):
            graph_proj = self.graph_projection(processed_modality_reprs["graph"])
            final_individual_representations["graph"] = graph_proj

        # Step 4: Fusion adaptative par gate mechanism using final_individual_representations (all in output_dim)
        combined_input_for_gate = torch.cat(
            [text_proj, image_proj, audio_proj, video_proj, graph_proj], dim=-1
        )
        # Ensure fusion_gate input matches expected (output_dim * 5)
        if combined_input_for_gate.shape[1] != self.output_dim * 5:
             # This might happen if some modalities are None and represented by zeros.
             # The fusion_gate is nn.Linear(output_dim * 5, ...), so it expects this size.
             # This implies the zero tensors must be correctly sized output_dim.
             pass # Already handled by initializing *_proj with zeros of output_dim

        fusion_weights_gate = self.fusion_gate(combined_input_for_gate)
        
        fused_repr = (
            fusion_weights_gate[..., 0:1] * text_proj +
            fusion_weights_gate[..., 1:2] * image_proj +
            fusion_weights_gate[..., 2:3] * audio_proj +
            fusion_weights_gate[..., 3:4] * video_proj +
            fusion_weights_gate[..., 4:5] * graph_proj
        )
        
        if return_individual_modalities:
            return fused_repr, final_individual_representations
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
        elif "video" in inputs and inputs["video"] is not None:
            return inputs["video"].size(0)
        elif "graph" in inputs and inputs["graph"] is not None:
            node_features = inputs["graph"].get("node_features")
            if node_features is not None:
                return node_features.size(0)
        
        # Valeur par défaut
        return 1
    
    def _get_device(self, inputs: Dict[str, Union[List[str], torch.Tensor, None]]) -> torch.device:
        """Détermine le device à partir des entrées"""
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and value is not None:
                return value.device
        
        # Valeur par défaut
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultimodalGeneration(nn.Module):
    """
    Module de génération multimodale pour NeuroLite.
    Permet de générer des sorties dans différentes modalités à partir
    d'une représentation latente commune.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 768,
        dropout_rate: float = 0.1,
        vocab_size: int = 50000,
        image_size: int = 224,
        audio_sample_rate: int = 16000,
        audio_length_ms: int = 30000,
        video_frames: int = 16,
        max_graph_nodes: int = 32
    ):
        """
        Initialise le module de génération multimodale.
        
        Args:
            input_dim: Dimension d'entrée (latent)
            hidden_dim: Dimension cachée interne
            dropout_rate: Taux de dropout
            vocab_size: Taille du vocabulaire pour le texte
            image_size: Taille des images générées (supposées carrées)
            audio_sample_rate: Taux d'échantillonnage audio
            audio_length_ms: Durée de l'audio généré en millisecondes
            video_frames: Nombre de trames dans les vidéos générées
            max_graph_nodes: Nombre maximal de nœuds dans un graphe généré
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Décodeurs spécialisés
        self.text_decoder = TextDecoder(
            input_dim=input_dim,
            vocab_size=vocab_size,
            embedding_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            num_layers=3,
            num_heads=8,
            dropout_rate=dropout_rate
        )
        
        self.image_decoder = ImageDecoder(
            input_dim=input_dim,
            output_channels=3,
            initial_channels=hidden_dim // 2,
            output_size=image_size,
            dropout_rate=dropout_rate,
            final_activation="tanh"
        )
        
        # Calculer une longueur de sortie audio qui est divisible par 512 (2^5 * 16)
        # Le décodeur audio a besoin que output_length soit divisible par (2^num_upsamples * initial_length)
        raw_output_length = audio_sample_rate * audio_length_ms // 1000  # 480000 pour 16000Hz * 30s
        upsampling_factor = 512  # 2^5 (num_upsamples=5) * 16 (initial_length)
        adjusted_output_length = (raw_output_length // upsampling_factor) * upsampling_factor
        # Si la longueur ajustée est trop différente, ajouter un bloc pour obtenir la division exacte
        if adjusted_output_length < raw_output_length:
            adjusted_output_length += upsampling_factor
        
        self.audio_decoder = AudioDecoder(
            input_dim=input_dim,
            initial_channels=hidden_dim // 2,
            initial_length=16,  # Défaut dans le décodeur
            output_channels=1,
            output_length=adjusted_output_length,  # Longueur ajustée pour être divisible par 512
            num_upsamples=5,  # Valeur par défaut dans le décodeur
            dropout_rate=dropout_rate,
            final_activation="tanh"
        )
        
        # Ajustement pour le décodeur vidéo
        # La valeur par défaut pour temporal_upsampling est 16 (2^4)
        # et initial_time est 1, donc nous devons nous assurer que output_time est un multiple de 16
        temporal_upsampling_factor = 16
        adjusted_video_frames = (video_frames // temporal_upsampling_factor) * temporal_upsampling_factor
        if adjusted_video_frames < video_frames:
            adjusted_video_frames += temporal_upsampling_factor
        # Garantir au moins 16 frames
        adjusted_video_frames = max(adjusted_video_frames, temporal_upsampling_factor)
        
        self.video_decoder = VideoDecoder(
            input_dim=input_dim,
            initial_channels=hidden_dim // 2,
            output_channels=3,
            output_time=adjusted_video_frames,  # Nombre de frames ajusté
            output_size=image_size,
            initial_time=1,  # Valeur par défaut
            dropout_rate=dropout_rate,
            final_activation="tanh"
        )
        
        self.graph_decoder = GraphDecoder(
            input_dim=input_dim,
            node_feature_dim=64,
            hidden_dim=hidden_dim,
            max_nodes=max_graph_nodes,
            num_layers=3,
            num_heads=8,
            dropout_rate=dropout_rate
        )
        
        # Projections de l'espace latent vers les espaces spécifiques des modalités
        self.to_text_latent = nn.Linear(input_dim, input_dim)
        self.to_image_latent = nn.Linear(input_dim, input_dim)
        self.to_audio_latent = nn.Linear(input_dim, input_dim)
        self.to_video_latent = nn.Linear(input_dim, input_dim)
        self.to_graph_latent = nn.Linear(input_dim, input_dim)
    
    def forward(
        self,
        latent: torch.Tensor,
        target_modalities: List[str] = None,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Génère des sorties dans différentes modalités à partir d'une représentation latente.
        
        Args:
            latent: Représentation latente [batch_size, input_dim]
            target_modalities: Liste des modalités cibles à générer
            text_input_ids: IDs des tokens d'entrée pour la génération de texte conditionnelle
            text_attention_mask: Masque d'attention pour text_input_ids
            temperature: Contrôle la variabilité des sorties générées
            
        Returns:
            Dictionnaire contenant les sorties générées pour chaque modalité demandée
        """
        batch_size = latent.size(0)
        results = {}
        
        # Générer pour chaque modalité demandée
        modalities = target_modalities or ["text", "image", "audio", "video", "graph"]
        
        if "text" in modalities:
            text_latent = self.to_text_latent(latent)
            
            if text_input_ids is not None:
                # Génération conditionnelle (teacher forcing)
                text_logits = self.text_decoder(text_latent, text_input_ids, text_attention_mask)
                results["text_logits"] = text_logits
            else:
                # Génération auto-régressive
                generated_text = self.text_decoder.generate(
                    text_latent,
                    temperature=temperature,
                    max_length=100,
                    do_sample=True
                )
                results["generated_text"] = generated_text
        
        if "image" in modalities:
            image_latent = self.to_image_latent(latent)
            generated_image = self.image_decoder.generate(
                image_latent,
                temperature=temperature
            )
            results["generated_image"] = generated_image
        
        if "audio" in modalities:
            audio_latent = self.to_audio_latent(latent)
            generated_audio = self.audio_decoder.generate(
                audio_latent,
                temperature=temperature
            )
            results["generated_audio"] = generated_audio
        
        if "video" in modalities:
            video_latent = self.to_video_latent(latent)
            generated_video = self.video_decoder.generate(
                video_latent,
                temperature=temperature
            )
            results["generated_video"] = generated_video
        
        if "graph" in modalities:
            graph_latent = self.to_graph_latent(latent)
            generated_graph = self.graph_decoder.generate(
                graph_latent,
                temperature=temperature
            )
            results["generated_graph"] = generated_graph
        
        return results


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
        batch_size = query_modality.size(0)
        
        # Normalisation de couche
        q = self.layer_norm(query_modality)
        
        # Projections
        q = self.q_proj(q)
        k = self.k_proj(key_value_modality)
        v = self.v_proj(key_value_modality)
        
        # Reshaper pour l'attention multi-têtes
        q = q.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        
        # Calcul de l'attention
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_size ** 0.5)
        
        # Appliquer le masque d'attention si fourni
        if attention_mask is not None:
            # Adapter le masque au format [batch, num_heads, seq_len_q, seq_len_kv]
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            
            # Masquer avec une valeur très négative
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Softmax et dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Appliquer les poids d'attention aux valeurs
        context = torch.matmul(attn_weights, v)
        
        # Reshaper et projeter
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        output = self.output_proj(context)
        
        # Connexion résiduelle
        output = output + query_modality
        
        return output