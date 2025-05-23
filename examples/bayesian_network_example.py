"""
Example demonstrating the setup and use of NeuroLiteModel with the BayesianBeliefNetwork.
This example shows how to configure the model to include the probabilistic reasoning component.
"""

import torch
from neurolite.model import NeuroLiteModel
from neurolite.config import NeuroLiteConfig
from neurolite.symbolic import BayesianBeliefNetwork # For type checking

def run_bayesian_network_example():
    print("Starting Bayesian Network Example...")

    # 1. Configure the model to use the BayesianBeliefNetwork.
    # We'll start with a small configuration and enable the Bayesian module.
    config = NeuroLiteConfig.small() # Using small for variety, could be tiny() or base()
    
    config.use_bayesian_module = True
    config.num_bayesian_variables = 3
    # Define a simple chain structure: Var0 -> Var1 -> Var2
    config.bayesian_network_structure = [(0, 1), (1, 2)] 
    # max_parents_bayesian is used if structure is None, but good to be aware of
    config.max_parents_bayesian = 1 

    print(f"\nNeuroLite Configuration (Bayesian):")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Bayesian Module Enabled: {config.use_bayesian_module}")
    print(f"  Number of Bayesian Variables: {config.num_bayesian_variables}")
    print(f"  Bayesian Network Structure: {config.bayesian_network_structure}")

    # 2. Instantiate NeuroLiteModel with this configuration.
    # Using task_type='base' for general demonstration.
    try:
        # The NeuroLiteModel's __init__ needs to be updated to pass the 'config' object
        # to the BayesianBeliefNetwork if it's not already doing so.
        # Assuming NeuroLiteModel passes its own config to its components.
        model = NeuroLiteModel(config, task_type="base")
        model.eval() # Set to evaluation mode
        print("\nNeuroLiteModel instantiated successfully with Bayesian configuration.")
    except Exception as e:
        print(f"Error instantiating NeuroLiteModel: {e}")
        # This might fail if NeuroLiteModel's __init__ doesn't pass 'config' to BayesianBeliefNetwork
        # or if BayesianBeliefNetwork expects 'config' and doesn't get it.
        # The provided BayesianBeliefNetwork __init__ was updated to accept 'config'.
        # The NeuroLiteModel __init__ needs to be checked/updated to pass it.
        # For now, this script assumes NeuroLiteModel correctly initializes BayesianBeliefNetwork with config.
        return

    # Verify if the BayesianBeliefNetwork is part of the model structure
    # This depends on how NeuroLiteModel stores its components.
    # Assuming it's stored in an attribute like `model.bayesian_network` or similar.
    # The current NeuroLiteModel does not have a dedicated attribute `bayesian_network`.
    # It seems the BayesianBeliefNetwork is intended to be one of the layers in `model.layers`
    # or a separate component similar to `model.symbolic`.
    # For this example, let's assume it might be added to `model.layers` or a new attribute.
    # We will search for it in model.layers for this example.
    
    bayesian_layer_present = False
    if hasattr(model, 'layers'): # Common way to store layers
        for layer in model.layers:
            if isinstance(layer, BayesianBeliefNetwork):
                bayesian_layer_present = True
                break
    
    if hasattr(model, 'bayesian_belief_network') and isinstance(model.bayesian_belief_network, BayesianBeliefNetwork):
        # This assumes NeuroLiteModel might store it as 'bayesian_belief_network'
        bayesian_layer_present = True

    # The previous structure of NeuroLiteModel's __init__ does not explicitly add BayesianBeliefNetwork.
    # This example assumes that NeuroLiteModel would be modified to include it, e.g.:
    # if config.use_bayesian_module:
    #     self.bayesian_network = BayesianBeliefNetwork(config, config.hidden_size, ...)
    #     # And then ensure it's called in the forward pass.
    # For now, we print a placeholder message.
    if bayesian_layer_present:
         print("BayesianBeliefNetwork is present in the model.")
         # We can also check its configured structure if accessible
         # network_instance = model.bayesian_network # or find in layers
         # print(f"  Network structure loaded: {network_instance.parents_matrix}")
    else:
        print("BayesianBeliefNetwork is NOT present or not found in expected attributes.")
        print("This example requires NeuroLiteModel to be updated to instantiate and use BayesianBeliefNetwork.")


    # 3. Create a sample batch of input texts.
    sample_texts = [
        "The weather is cloudy and the barometer is falling.",
        "Patient has a fever and a cough.",
        "Stock prices are volatile due to market uncertainty."
    ]
    print(f"\nSample input texts: {sample_texts}")

    # 4. Pass the texts through the model.
    print("\nProcessing input texts through the model...")
    try:
        with torch.no_grad():
            outputs = model(input_texts=sample_texts)
        
        if isinstance(outputs, dict) and "hidden_states" in outputs:
            final_hidden_states = outputs["hidden_states"]
            print(f"Model processing complete. Output hidden states shape: {final_hidden_states.shape}")
        else:
            print(f"Model processing complete. Output type: {type(outputs)}")

        if bayesian_layer_present:
            print("\nBayesian Network Processing Notes:")
            print("- If the BayesianBeliefNetwork was activated, it would have performed these steps:")
            print("  1. Evidence Extraction: Extracted probabilities for Bayesian variables from neural states.")
            print("  2. Probabilistic Inference: Used an algorithm (e.g., Likelihood Weighting) "
                  "to estimate posterior probabilities of variables given the evidence and network structure.")
            print("  3. Integration: The inferred probabilities would be combined back into the neural hidden states.")
            print("- The final output hidden_states have potentially been influenced by this probabilistic reasoning.")

    except Exception as e:
        print(f"Error during model forward pass: {e}")

    print("\nBayesian Network Example finished.")

if __name__ == "__main__":
    run_bayesian_network_example()
    # To run this example:
    # 1. Ensure NeuroLiteModel's __init__ is updated to create an instance of BayesianBeliefNetwork
    #    when config.use_bayesian_module is True.
    # 2. Ensure NeuroLiteModel's forward pass calls this BayesianBeliefNetwork instance.
    # Example: python examples/bayesian_network_example.py
