import torch
import json
import os
from typing import Optional, Dict, List, Tuple, Any, Union

# Adjust import path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurolite.core.model import NeuroLiteModel
from neurolite.Configs.config import NeuroLiteConfig
from neurolite.memory.hierarchical_memory import HierarchicalMemory
from neurolite.reasoning.reasoning import NeurosymbolicReasoner

# --- Configuration ---
DEVICE = "cpu"
SYMBOLIC_DIM = 64
DUMMY_RULES_FILE = "dummy_rules_advanced.json"

def create_dummy_rules_file(filepath: str = DUMMY_RULES_FILE):
    rules_content = {
        "facts": [{"predicate": "isa", "args": ["cat", "animal"]}, {"predicate": "isa", "args": ["man", "mortal"]}],
        "rules": [{"head": {"predicate": "mortal", "args": ["?x"]}, "body": [{"predicate": "isa", "args": ["?x", "man"]}]}]
    }
    try:
        with open(filepath, 'w') as f:
            json.dump(rules_content, f, indent=2)
        print(f"Created dummy rules file: {filepath}")
    except IOError as e:
        print(f"Error creating dummy rules file {filepath}: {e}")

def initialize_model_and_memory() -> NeuroLiteModel:
    print("\n--- Initializing Model with Memory and Reasoner ---")
    config = NeuroLiteConfig.base()
    print(f"Initial config.device from NeuroLiteConfig.base(): {config.device}")

    config.device = DEVICE
    print(f"Overridden config.device to: {config.device}")

    # Ensure sub-configs also reflect CPU if they have device settings (they usually don't, NeuroLiteConfig.device is primary)
    if hasattr(config.model_config, 'device') and config.model_config.device != DEVICE:
        print(f"Warning: model_config.device was {config.model_config.device}, setting to {DEVICE}")
        config.model_config.device = DEVICE
    if hasattr(config.memory_config, 'device') and config.memory_config.device != DEVICE: # memory_config doesn't have device
        pass


    config.model_config.use_external_memory = True
    config.model_config.use_hierarchical_memory = True
    config.model_config.use_symbolic_module = True
    config.model_config.use_advanced_reasoning = True
    config.model_config.symbolic_dim = SYMBOLIC_DIM
    config.model_config.symbolic_rules_file = DUMMY_RULES_FILE

    print(f"Final config.device before NeuroLiteModel init: {config.device}")
    print(f"Model config: use_multimodal_input = {config.model_config.use_multimodal_input}")
    print(f"Model config: input_projection_type = {config.model_config.input_projection_type}")

    model = NeuroLiteModel(config, task_type="base")
    model.eval()
    # model.to(DEVICE) # NeuroLiteModel's __init__ should handle device placement based on config.device

    print("NeuroLiteModel initialized.")
    print(f"Model device after init: {next(model.parameters()).device}") # Check actual model parameter device
    assert isinstance(model.memory, HierarchicalMemory), "HierarchicalMemory not initialized correctly."
    assert isinstance(model.symbolic, NeurosymbolicReasoner), "NeurosymbolicReasoner not initialized."
    print(f"Memory type: {type(model.memory)}")
    print(f"Symbolic module type: {type(model.symbolic)}")

    return model

def run_inference_step(model: NeuroLiteModel,
                       text_prompt: str,
                       external_facts_tensor: Optional[torch.Tensor] = None,
                       step_name: str = "Inference Step") -> tuple[Optional[Dict[str, Any]], Optional[torch.Tensor]]:
    print(f"\n--- {step_name} ---")
    print(f"Input prompt: '{text_prompt}'")

    if external_facts_tensor is not None:
        print(f"Using external_facts_tensor of shape: {external_facts_tensor.shape}")

    multimodal_inputs = {"text": [text_prompt]}

    print("(Conceptual: NeurosymbolicReasoner.forward will attempt to read from its memory_system if available)")
    outputs = model.forward(
        multimodal_inputs=multimodal_inputs,
        external_facts=external_facts_tensor,
        return_symbolic=True,
        output_hidden_states=False
    )
    print("(Conceptual: NeurosymbolicReasoner.forward may attempt to write important conclusions to its memory_system)")

    symbolic_outputs = outputs.get("symbolic_outputs")
    final_hidden_states = outputs.get("hidden_states")

    if symbolic_outputs:
        print("Symbolic Outputs:")
        for key, value in symbolic_outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor of shape {value.shape}")
            else:
                print(f"  {key}: {value}")
    else:
        print("No symbolic_outputs returned by the model.")

    if final_hidden_states is not None:
        print(f"Final hidden_states shape: {final_hidden_states.shape}")

    return symbolic_outputs, final_hidden_states

def main():
    create_dummy_rules_file()
    model = initialize_model_and_memory()

    initial_external_fact_tensor = torch.randn(1, 1, SYMBOLIC_DIM, device=DEVICE)
    print("\nSimulating initial fact 'All men are mortal' as a tensor.")

    symbolic_outputs_step1, _ = run_inference_step(
        model,
        text_prompt="Socrates is a man.",
        external_facts_tensor=initial_external_fact_tensor,
        step_name="Step 1: Socrates is a man + All men are mortal"
    )

    print("\n--- Step 2: Conceptual Conclusion Processing ---")
    if symbolic_outputs_step1 and symbolic_outputs_step1.get('relations') is not None:
        derived_fact_tensor = torch.randn(1, 1, SYMBOLIC_DIM, device=DEVICE)
        print("Simulating derived fact 'Socrates is mortal' as a new tensor for next step.")
        print("(This simulates either a conclusion stored/retrieved from memory, or passed as new external_facts)")
    else:
        print("Skipping derived fact simulation as prior symbolic outputs are missing or 'relations' key not found.")
        derived_fact_tensor = None

    symbolic_outputs_step2, _ = run_inference_step(
        model,
        text_prompt="Is Socrates mortal?",
        external_facts_tensor=derived_fact_tensor,
        step_name="Step 3: Query about Socrates' mortality (with derived fact)"
    )

    print("\n--- Step 4: Conceptual Memory Verification ---")
    if model.memory:
        print("HierarchicalMemory was active. During NeurosymbolicReasoner's execution:")
        print("  - It could have performed searches in its memory_system (STM, LTM, PM).")
        print("  - It could have stored new important conclusions (e.g., 'Socrates is mortal') in its memory_system.")
        print("Actual verification of memory content would require a more sophisticated setup and specific search queries.")

    if os.path.exists(DUMMY_RULES_FILE):
        try:
            os.remove(DUMMY_RULES_FILE)
            print(f"\nCleaned up dummy rules file: {DUMMY_RULES_FILE}")
        except IOError as e:
            print(f"Error cleaning up dummy rules file {DUMMY_RULES_FILE}: {e}")

if __name__ == "__main__":
    main()
