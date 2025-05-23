"""
Example demonstrating the setup and use of NeuroLiteModel with the NeuralSymbolicLayer.
This example shows how to configure the model to include the symbolic reasoning component.
"""

import torch
from neurolite.model import NeuroLiteModel
from neurolite.config import NeuroLiteConfig
from neurolite.symbolic import NeuralSymbolicLayer # For type checking
import os

def run_symbolic_reasoning_example():
    print("Starting Symbolic Reasoning Example...")

    # 1. Configure the model to use symbolic reasoning.
    # The base_symbolic() class method provides a default configuration
    # with the symbolic module enabled and pointing to "rules.json".
    config = NeuroLiteConfig.base_symbolic()

    # Ensure the default rules file "rules.json" exists.
    # This example assumes "rules.json" is in the project root or a location
    # accessible by the model. If NeuroLiteModel is typically run from the project root,
    # "rules.json" should be found.
    # For robustness, you might want to ensure its creation or provide a path.
    # Here, we rely on the `base_symbolic` config default.
    rules_file_path = config.symbolic_rules_file
    if not os.path.exists(rules_file_path):
        print(f"Warning: Default rules file '{rules_file_path}' not found.")
        print("Please ensure 'rules.json' (as created in previous steps) is in the project root.")
        # As a fallback for this example, let's create a minimal dummy one if it's missing,
        # so the example can run without manual file creation for demonstration.
        print(f"Creating a minimal dummy '{rules_file_path}' for demonstration purposes.")
        dummy_rules = {
            "facts": ["is_a(cat, animal)"],
            "rules": [{"premises": ["is_a(?x, animal)"], "conclusion": "has_dna(?x)"}]
        }
        try:
            with open(rules_file_path, 'w') as f:
                import json
                json.dump(dummy_rules, f)
            print(f"Dummy '{rules_file_path}' created successfully.")
        except Exception as e:
            print(f"Error creating dummy rules file: {e}. Symbolic layer might not function as expected.")
            # Set to None if creation fails, so SymbolicRuleEngine doesn't try to load a non-existent file.
            config.symbolic_rules_file = None 


    print(f"\nNeuroLite Configuration (Symbolic):")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Symbolic Module Enabled: {config.use_symbolic_module}")
    print(f"  Symbolic Rules File: {config.symbolic_rules_file}")
    print(f"  Max Predicate Types: {getattr(config, 'max_predicate_types', 'N/A')}") # Added in later steps
    print(f"  Max Entities in Vocab: {getattr(config, 'max_entities_in_vocab', 'N/A')}") # Added in later steps


    # 2. Instantiate NeuroLiteModel with this configuration.
    # We use task_type='base' as we are not focusing on a specific downstream task here.
    try:
        model = NeuroLiteModel(config, task_type="base")
        model.eval() # Set to evaluation mode for consistent behavior (e.g. dropout, symbolic layer activation)
        print("\nNeuroLiteModel instantiated successfully with symbolic configuration.")
    except Exception as e:
        print(f"Error instantiating NeuroLiteModel: {e}")
        return

    # Verify if the NeuralSymbolicLayer is part of the model structure
    symbolic_layer_present = False
    if model.symbolic is not None and isinstance(model.symbolic, NeuralSymbolicLayer):
        symbolic_layer_present = True
        print("NeuralSymbolicLayer is present in the model.")
        if model.symbolic.rule_engine:
            print(f"  SymbolicRuleEngine loaded with {len(model.symbolic.rule_engine.facts.keys())} initial fact predicates.")
            print(f"  SymbolicRuleEngine loaded with {len(model.symbolic.rule_engine.rules)} rules.")
    else:
        print("NeuralSymbolicLayer is NOT present in the model. Check config.use_symbolic_module.")


    # 3. Create a sample batch of input texts.
    # The actual content of the texts might influence entity/relation extraction if the
    # extraction mechanisms were more sophisticated. For this example, the focus is on setup.
    sample_texts = [
        "The cat sat on the mat. It was a fluffy cat.",
        "A dog chased a squirrel up a tree.",
        "Symbolic reasoning can enhance neural models."
    ]
    print(f"\nSample input texts: {sample_texts}")

    # 4. Pass the texts through the model.
    # The model's forward pass will internally invoke the NeuralSymbolicLayer
    # if it's configured and conditions for its activation are met (e.g., if not self.training).
    print("\nProcessing input texts through the model...")
    try:
        # Using input_texts as per NeuroLiteModel's capability
        # For models expecting token IDs, ensure tokenizer and input_ids are used.
        # The base NeuroLiteModel can accept input_texts.
        with torch.no_grad(): # Disable gradient calculations for inference
            outputs = model(input_texts=sample_texts)
        
        # The 'outputs' will typically be a dictionary. For a 'base' model, it contains 'hidden_states'.
        if isinstance(outputs, dict) and "hidden_states" in outputs:
            final_hidden_states = outputs["hidden_states"]
            print(f"Model processing complete. Output hidden states shape: {final_hidden_states.shape}")
        else:
            print(f"Model processing complete. Output type: {type(outputs)}")

        if symbolic_layer_present:
            print("\nSymbolic Processing Notes:")
            print("- If the NeuralSymbolicLayer was activated (e.g., in eval mode or by chance during training), "
                  "it would have performed the following conceptual steps:")
            print("  1. Entity Extraction: Identified potential entities from the input's neural representations.")
            print("  2. Relation Extraction: Deduced potential relations between these entities.")
            print("  3. Symbolic Inference: Added these extracted facts/relations to its rule engine "
                  "(along with any pre-loaded rules from 'rules.json') and derived new facts.")
            print("  4. Integration: The information from derived facts (represented by learnable embeddings "
                  "for predicate types and entity IDs) would be combined with the neural hidden states.")
            print("- The final output hidden_states have potentially been influenced by this symbolic reasoning process.")
            print("- To see concrete derived facts, one would typically need to inspect the "
                  "NeuralSymbolicLayer's internal state or have a task output that directly reflects them.")
        
        # Example: Check if vocabularies in the symbolic layer have grown (if new predicates/entities were found)
        if symbolic_layer_present and model.symbolic:
            print("\nSymbolic Layer Vocabulary (after processing):")
            print(f"  Predicate vocabulary size: {len(model.symbolic.predicate_to_idx)}")
            print(f"  Symbolic entity vocabulary size: {len(model.symbolic.symbolic_entity_to_idx)}")
            # print(f"  Predicates: {list(model.symbolic.predicate_to_idx.keys())}") # Potentially long
            # print(f"  Entities: {list(model.symbolic.symbolic_entity_to_idx.keys())}") # Potentially very long


    except Exception as e:
        print(f"Error during model forward pass: {e}")

    print("\nSymbolic Reasoning Example finished.")

if __name__ == "__main__":
    run_symbolic_reasoning_example()
    # To run this example, ensure you are in the root directory of the NeuroLite project
    # and that "rules.json" (or the dummy one created) is accessible.
    # Example: python examples/symbolic_reasoning_example.py
