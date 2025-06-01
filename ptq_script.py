import torch
import torch.quantization
import os
import sys
import time # For temp file naming

# Add neurolite to Python path if not installed (adjust path as needed for sandbox)
# Assuming the script is run from /app and neurolite is in /app/neurolite
sys.path.insert(0, '/app')

from neurolite.core.model import NeuroLiteModel
from neurolite.Configs.config import NeuroLiteConfig

def get_model_size(model, filename_prefix="temp_model"):
    """Calculates model size by saving state_dict to a temporary file."""
    # Ensure model is on CPU before saving for consistent size measurement
    model_to_save = model.cpu()
    temp_filename = f"{filename_prefix}_{time.time_ns()}.pt"
    torch.save(model_to_save.state_dict(), temp_filename)
    size_bytes = os.path.getsize(temp_filename)
    os.remove(temp_filename)
    return size_bytes / (1024 * 1024) # Convert to MB

def main():
    print("Starting PTQ exploration script...")

    # 1. Initialize Model
    print("\n1. Initializing Model...")
    try:
        config = NeuroLiteConfig.base()

        # Override device to CPU for this CPU-only environment
        config.device = "cpu"
        # Also ensure relevant sub-configs reflect this if they have device settings
        config.model_config.device = "cpu" # Assuming ModelArchitectureConfig might have a device field
                                           # (It doesn't, but NeuroLiteConfig.device is the main one)

        model_arch_config = config.model_config # For checks later if needed

        model = NeuroLiteModel(config, task_type="base")
        model.eval() # Set to evaluation mode
        model.to(config.device) # Explicitly move model to CPU

        print("NeuroLiteModel initialized successfully on CPU.")
        print(f"Model uses multimodal input: {model.model_config.use_multimodal_input}")

    except Exception as e:
        print(f"Error during model initialization: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Calculate Original Model Size
    print("\n2. Calculating Original Model Size...")
    try:
        original_size_mb = get_model_size(model, "original_neurolite")
        print(f"Original model size: {original_size_mb:.2f} MB")
    except Exception as e:
        print(f"Error calculating original model size: {e}")
        if 'model' not in locals():
            return

    # 3. Post-Training Dynamic Quantization
    print("\n3. Applying Post-Training Dynamic Quantization...")
    quantized_model = None
    try:
        layers_to_quantize = {torch.nn.Linear}
        # Ensure model is on CPU before quantization for dynamic quantization
        model.to("cpu")
        quantized_model = torch.quantization.quantize_dynamic(
            model, layers_to_quantize, dtype=torch.qint8
        )
        print("Dynamic quantization applied.")
    except Exception as e:
        print(f"Error during dynamic quantization: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Calculate Quantized Model Size
    print("\n4. Calculating Quantized Model Size...")
    if quantized_model:
        try:
            # Quantized model is already on CPU
            quantized_size_mb = get_model_size(quantized_model, "quantized_neurolite")
            print(f"Quantized model size: {quantized_size_mb:.2f} MB")
            if 'original_size_mb' in locals() and original_size_mb > 0 :
                reduction = original_size_mb - quantized_size_mb
                reduction_percent = (reduction / original_size_mb) * 100
                print(f"Size reduction: {reduction:.2f} MB ({reduction_percent:.2f}%)")
            elif 'original_size_mb' in locals():
                 print(f"Original model size was {original_size_mb:.2f} MB, cannot calculate reduction percentage.")
        except Exception as e:
            print(f"Error calculating quantized model size: {e}")

    # 5. Simple Inference Test
    print("\n5. Performing Simple Inference Test on Quantized Model...")
    if quantized_model:
        try:
            dummy_input_dict = {}
            # NeuroLiteConfig.base() by default has model_config.use_multimodal_input = False
            # and input_projection_type = "minhash_bloom"
            if model.model_config.use_multimodal_input:
                 dummy_input_dict = {
                    "text": ["This is a test sentence for NeuroLite after quantization."]
                 }
            else: # Not using multimodal input
                if model.model_config.input_projection_type == "minhash_bloom":
                    dummy_input_dict = {"text": ["Test sentence for MinHashBloom."]}
                else: # Assumes "tokenized_minhash" or similar needing input_ids
                    dummy_input_ids = torch.randint(0, model.model_config.vocab_size, (1, 10), device=config.device)
                    dummy_attention_mask = torch.ones_like(dummy_input_ids, device=config.device)
                    dummy_input_dict = {"input_ids": dummy_input_ids, "attention_mask": dummy_attention_mask}

            print(f"Using dummy input keys: {dummy_input_dict.keys()}")

            with torch.no_grad():
                # Ensure dummy input tensors are on CPU if model is CPU
                for key, value in dummy_input_dict.items():
                    if isinstance(value, torch.Tensor):
                        dummy_input_dict[key] = value.to(config.device)

                output = quantized_model.forward(multimodal_inputs=dummy_input_dict)

            if isinstance(output, dict) and "hidden_states" in output:
                print(f"Inference successful. Output hidden_states shape: {output['hidden_states'].shape}")
                print(f"Output device: {output['hidden_states'].device}")
            else:
                print(f"Inference produced output, but structure might be unexpected: {type(output)}")
                if torch.is_tensor(output):
                    print(f"Output tensor shape: {output.shape}, device: {output.device}")
                elif isinstance(output, dict):
                     print(f"Output dict keys: {output.keys()}")

        except Exception as e:
            print(f"Error during inference with quantized model: {e}")
            import traceback
            traceback.print_exc()

    print("\nPTQ exploration script finished.")

if __name__ == "__main__":
    main()
