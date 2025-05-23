"""
Example demonstrating the use of NeuroLiteModel with multimodal inputs.

This script shows how to:
1. Configure NeuroLiteModel for multimodal input processing.
2. Prepare dummy data for text, image, audio, and video modalities.
3. Pass these inputs to the model.
4. Inspect the output, including individual modality representations if cross-modal attention is enabled.
"""

import torch
from neurolite.model import NeuroLiteModel
from neurolite.config import NeuroLiteConfig
from neurolite.multimodal import MultimodalProjection # For type checking if needed

def run_multimodal_input_example():
    print("Starting Multimodal Input Example...")

    # 1. Configuration
    print("\n--- Configuring Model ---")
    config = NeuroLiteConfig.tiny()  # Start with a tiny config for efficiency

    # Enable multimodal input processing
    config.use_multimodal_input = True

    # Set related multimodal parameters
    # If multimodal_output_dim is 0 or matches hidden_size, no extra projection is added after fusion.
    # Let's set it to a different value to test the projection layer if it exists.
    # config.multimodal_output_dim = config.hidden_size // 2 # Example: different output dim
    config.multimodal_output_dim = config.hidden_size # Example: same output dim, no final projection needed for fused output
    
    config.multimodal_image_patch_size = 16 # Default, but explicit for example
    config.multimodal_video_num_sampled_frames = 3 # Reduced for faster dummy data, default is 5

    # Optionally, enable cross-modal attention to demonstrate returning individual modalities
    config.use_cross_modal_attention = True # Set to True to get individual_modality_representations
    config.cross_modal_num_heads = 2 # Default is 4, using 2 for tiny config

    print(f"Configuration set:")
    print(f"  Use Multimodal Input: {config.use_multimodal_input}")
    print(f"  Multimodal Output Dimension (Projection Layer): {config.multimodal_output_dim if config.multimodal_output_dim > 0 else config.hidden_size}")
    print(f"  Image Patch Size: {config.multimodal_image_patch_size}")
    print(f"  Video Sampled Frames: {config.multimodal_video_num_sampled_frames}")
    print(f"  Use Cross-Modal Attention: {config.use_cross_modal_attention}")
    print(f"  Cross-Modal Attention Heads: {config.cross_modal_num_heads}")

    # 2. Model Instantiation
    print("\n--- Instantiating Model ---")
    model = NeuroLiteModel(config=config, task_type="base") # Using 'base' for general feature output
    model.eval()  # Set to evaluation mode
    print("NeuroLiteModel instantiated successfully.")

    # Verify that input_projection is indeed MultimodalProjection
    if isinstance(model.input_projection, MultimodalProjection):
        print("Model's input_projection is correctly set to MultimodalProjection.")
    else:
        print(f"Warning: Model's input_projection is {type(model.input_projection)}, not MultimodalProjection.")
    
    if config.use_cross_modal_attention:
        if model.cross_modal_attention_text_image is not None:
            print("CrossModalAttention module is present in the model.")
        else:
            print("Warning: CrossModalAttention module is NOT present despite config.")


    # 3. Prepare Dummy Multimodal Inputs
    print("\n--- Preparing Dummy Multimodal Inputs ---")
    batch_size = 2

    # Text Input
    dummy_text = ["A sample text document for NeuroLite.", "Another piece of text for processing."]
    if len(dummy_text) != batch_size: dummy_text = dummy_text[:batch_size] # Adjust if needed
    print(f"Dummy Text (Batch Size {len(dummy_text)}): {dummy_text}")

    # Image Input (Batch, Channels, Height, Width)
    # MultimodalProjection's image_encoder expects LayerNorm([16, 14, 14]) after Conv2d.
    # This implies input image size that results in 14x14 patches. If patch_size=16, 16*14 = 224.
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    print(f"Dummy Image shape: {dummy_image.shape}")

    # Audio Input (Batch, Channels, TimeSteps, FreqBins)
    # MultimodalProjection's audio_encoder expects LayerNorm([16, 64, 40]) after first Conv2d.
    # This gives a hint on expected input Mel-spectrogram dimensions.
    # Assuming input is [B, 1, T_audio, F_audio] where T_audio and F_audio are compatible.
    # E.g., if stride is 2, padding 1 for kernel 3x3, then T_audio=128, F_audio=80 could lead to 64x40.
    dummy_audio = torch.randn(batch_size, 1, 128, 80) 
    print(f"Dummy Audio shape: {dummy_audio.shape}")

    # Video Input (Batch, TotalFrames, Channels, Height, Width)
    # Make total frames more than sampled frames to test sampling logic.
    total_video_frames = config.multimodal_video_num_sampled_frames + 2 
    dummy_video = torch.randn(batch_size, total_video_frames, 3, 224, 224) # Assuming 224x224 frames like images
    print(f"Dummy Video shape: {dummy_video.shape} (Total frames: {total_video_frames}, Sampled: {config.multimodal_video_num_sampled_frames})")

    multimodal_inputs_full = {
        "text": dummy_text,
        "image": dummy_image,
        "audio": dummy_audio,
        "video": dummy_video
    }

    multimodal_inputs_partial = {
        "text": dummy_text,
        "image": dummy_image
        # Missing audio and video
    }
    
    multimodal_inputs_text_only = {
        "text": dummy_text
    }


    # 4. Model Forward Pass
    print("\n--- Performing Model Forward Pass ---")
    
    def process_and_print(inputs_dict, description):
        print(f"\n--- {description} ---")
        with torch.no_grad():
            outputs = model(multimodal_inputs=inputs_dict, return_dict=True)
        
        if "hidden_states" in outputs: # For base model
            print(f"Fused output 'hidden_states' shape: {outputs['hidden_states'].shape}")
            # Expected shape is [batch_size, 1, config.hidden_size] because MultimodalProjection's
            # fused output is unsqueezed to a sequence length of 1.
        elif "logits" in outputs: # For classification/generation models
             print(f"Output 'logits' shape: {outputs['logits'].shape}")
        else:
            print(f"Main output key not found. Available keys: {outputs.keys()}")

        if config.use_cross_modal_attention and "individual_modality_representations" in outputs:
            print("Individual Modality Representations:")
            for modality, tensor in outputs["individual_modality_representations"].items():
                print(f"  - {modality}: {tensor.shape}")
        elif config.use_cross_modal_attention:
            print("Cross-modal attention enabled, but 'individual_modality_representations' not in outputs.")

    # Test with full inputs
    process_and_print(multimodal_inputs_full, "Processing Full Multimodal Inputs")

    # Test with partial inputs
    process_and_print(multimodal_inputs_partial, "Processing Partial Multimodal Inputs (Text & Image)")
    
    # Test with text-only in multimodal mode (should still use MultimodalProjection)
    process_and_print(multimodal_inputs_text_only, "Processing Text-Only Input via MultimodalProjection")


    # Example of using the model in text-only mode (if use_multimodal_input was False)
    print("\n--- Conceptual: Text-Only Mode (if use_multimodal_input=False) ---")
    config_text_only = NeuroLiteConfig.tiny()
    config_text_only.use_multimodal_input = False # Explicitly disable multimodal
    model_text_only = NeuroLiteModel(config=config_text_only, task_type="base")
    model_text_only.eval()
    
    text_only_inputs_dict = {"text": dummy_text} # For _process_text_input via main forward
    # Or: text_only_inputs_dict = {"input_ids": torch.randint(0, config_text_only.vocab_size, (batch_size, 10))}

    with torch.no_grad():
        outputs_text_only = model_text_only(multimodal_inputs=text_only_inputs_dict, return_dict=True)
    
    if "hidden_states" in outputs_text_only:
        print(f"Text-only mode output 'hidden_states' shape: {outputs_text_only['hidden_states'].shape}")
        # Expected shape [batch_size, 1, config.hidden_size] for _process_text_input
        # or [batch_size, seq_len, config.hidden_size] for _process_token_input
    else:
        print(f"Text-only mode main output key not found. Keys: {outputs_text_only.keys()}")


    print("\nMultimodal Input Example finished.")

if __name__ == "__main__":
    run_multimodal_input_example()
    # To run this example:
    # Ensure that NeuroLiteModel and MultimodalProjection are correctly implemented
    # to handle the 'multimodal_inputs' dictionary and the 'return_individual_modalities' flag.
    # Example: python examples/multimodal_input_example.py
