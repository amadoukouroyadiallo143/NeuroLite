"""
Script pour initialiser le modèle NeuroLite et inspecter son architecture
et le nombre de ses paramètres.
"""
import torch

from neurolite.core.model import NeuroLiteModel
from neurolite.tokenization.tokenizer import NeuroLiteTokenizer
from neurolite.Configs.config import (
    NeuroLiteConfig,
    ModelArchitectureConfig, 
    TokenizerConfig, 
    MemoryConfig,
    ReasoningConfig,
    MMTextEncoderConfig,
    MMImageEncoderConfig,
    MMAudioEncoderConfig,
    MMVideoEncoderConfig,
    MMGraphEncoderConfig,
    MMTextDecoderConfig,
    MMImageDecoderConfig,
    MMAudioDecoderConfig,
    MMVideoDecoderConfig,
    MMGraphDecoderConfig
)

def count_parameters(model: torch.nn.Module):
    """Compte le nombre total et le nombre de paramètres entraînables d'un modèle."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def main():
    print("--- Initialisation du Modèle pour Inspection ---")

    # 1. Créer les configurations
    print("\nCréation des configurations pour le modèle et le tokenizer...")
    model_config = ModelArchitectureConfig(
        hidden_size=256, # Une taille raisonnable pour l'inspection
        # --- ACTIVATION DE TOUS LES MODULES ---
        use_metacontroller=True,
        use_hierarchical_memory=True,
        use_continual_adapter=True,
        # --- ACTIVATION DES ENCODEURS MULTIMODAUX ---
        mm_text_encoder_config=MMTextEncoderConfig(),
        mm_image_encoder_config=MMImageEncoderConfig(),
        mm_audio_encoder_config=MMAudioEncoderConfig(),
        mm_video_encoder_config=MMVideoEncoderConfig(),
        mm_graph_encoder_config=MMGraphEncoderConfig(),
        # --- ACTIVATION DES DECODEURS MULTIMODAUX ---
        mm_text_decoder_config=MMTextDecoderConfig(),
        mm_image_decoder_config=MMImageDecoderConfig(),
        mm_audio_decoder_config=MMAudioDecoderConfig(),
        mm_video_decoder_config=MMVideoDecoderConfig(),
        mm_graph_decoder_config=MMGraphDecoderConfig()
    )

    tokenizer_config = TokenizerConfig(
        hidden_size=model_config.hidden_size,
        # Les autres paramètres (codebook_size, num_quantizers) utilisent les valeurs par défaut
    )

    memory_config = MemoryConfig(
        use_external_memory=True,
        memory_dim=model_config.hidden_size
    )

    reasoning_config = ReasoningConfig(
        use_symbolic_module=True,
        use_causal_reasoning=True
    )

    config = NeuroLiteConfig(
        model_config=model_config,
        tokenizer_config=tokenizer_config,
        memory_config=memory_config,
        reasoning_config=reasoning_config
    )

    # 2. Instancier le tokenizer
    print("Instanciation du NeuroLiteTokenizer...")
    tokenizer = NeuroLiteTokenizer(tokenizer_config, neurolite_config=config)

    # 3. Instancier le modèle
    print("Instanciation du modèle NeuroLite...")
    try:
        # Le tokenizer est maintenant un argument requis pour le modèle
        # On spécifie le type de tâche pour voir les décodeurs
        model = NeuroLiteModel(config, task_type='multimodal_generation', tokenizer=tokenizer)
        print("Modèle instancié avec succès.")
    except Exception as e:
        print("\nERREUR : Échec de l'instanciation du modèle.")
        print(f"Détails de l'erreur : {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Compter et afficher les paramètres
    total_params, trainable_params = count_parameters(model)
    print("\n--- Nombre de Paramètres du Modèle ---")
    print(f"Paramètres totaux     : {total_params / 1_000_000:.2f} M")
    print(f"Paramètres entraînables : {trainable_params / 1_000_000:.2f} M")

    # 5. Afficher la structure du modèle pour inspection
    print("\n--- Architecture du Modèle ---")
    print(model)
    print("\n--- Inspection Terminée ---")

if __name__ == "__main__":
    main() 