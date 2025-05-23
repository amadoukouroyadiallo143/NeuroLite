"""
Lifelong Learning Demonstration for NeuroLite

This script demonstrates the conceptual use of ContinualAdapter and HierarchicalMemory
in a scenario where the model encounters data from different "tasks" or "domains" sequentially.

Task Concept:
The model processes text sequences in two phases:
1. Phase 1: "Animal Facts Domain" - The model sees sequences with facts about animals.
2. Phase 2: "Space Exploration Domain" - The topic shifts to space exploration.

The `ContinualAdapter` is intended to help the model adapt to the new domain (Phase 2)
by managing its experience buffer (e.g., replaying diverse samples, detecting drift).
The `HierarchicalMemory` (with its novelty-based consolidation and contextual gates)
is intended to manage knowledge from different phases, storing novel information
and potentially retrieving context-relevant memories.

This demo focuses on showing the components being active and processing data.
It does not involve actual model training or performance evaluation.
"""

import torch
from neurolite.model import NeuroLiteModel
from neurolite.config import NeuroLiteConfig
from neurolite.continual import ContinualAdapter # For type checking and buffer access
from neurolite.hierarchical_memory import HierarchicalMemory # For potential inspection

def run_lifelong_learning_demo():
    print("Starting Lifelong Learning Demonstration...\n")

    # 1. Configure the model
    config = NeuroLiteConfig.tiny() # Start with a tiny config for speed

    # Enable ContinualAdapter
    config.use_continual_adapter = True
    config.continual_adapter_buffer_size = 20 # Small buffer for demo
    config.continual_adapter_rate = 0.1      # Learning rate for adapter's internal model (if any)
    config.continual_adapter_drift_threshold = 0.5 # Conceptual threshold for drift detection

    # Enable HierarchicalMemory (use_external_memory is True by default in tiny())
    config.use_external_memory = True 
    # Set novelty thresholds for HierarchicalMemory's smarter consolidation
    # These values might need tuning in a real scenario.
    config.novelty_threshold_ltm = 0.55 # Update LTM if novelty is > 0.55
    config.novelty_threshold_pm = 0.65  # Update PM if novelty is > 0.65

    print("NeuroLite Configuration for Lifelong Learning:")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Continual Adapter Enabled: {config.use_continual_adapter}")
    print(f"  Continual Adapter Buffer Size: {config.continual_adapter_buffer_size}")
    print(f"  Continual Adapter Rate: {config.continual_adapter_rate}")
    print(f"  Hierarchical Memory Enabled: {config.use_external_memory}")
    print(f"  LTM Novelty Threshold: {config.novelty_threshold_ltm}")
    print(f"  PM Novelty Threshold: {config.novelty_threshold_pm}")
    print("-" * 40)

    # 2. Instantiate NeuroLiteModel
    try:
        model = NeuroLiteModel(config, task_type="base")
        model.eval() # Set to evaluation mode for consistent behavior
        print("NeuroLiteModel instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating NeuroLiteModel: {e}")
        return

    # Access the ContinualAdapter if present
    continual_adapter_module = model.continual_adapter if hasattr(model, 'continual_adapter') else None
    if continual_adapter_module:
        print(f"ContinualAdapter module found in model. Initial buffer size: {len(continual_adapter_module.buffer)}")
    else:
        print("Warning: ContinualAdapter module not found in model. Ensure 'use_continual_adapter' is True in config.")

    # Access HierarchicalMemory if present
    hierarchical_memory_module = model.memory if hasattr(model, 'memory') and isinstance(model.memory, HierarchicalMemory) else None
    if hierarchical_memory_module:
        print("HierarchicalMemory module found in model.")
    else:
        print("Warning: HierarchicalMemory module not found or not of expected type.")
    print("-" * 40)

    # 3. Simulate Data Phases

    # Phase 1 Data: Animal Facts
    phase1_data = [
        "Lions are large carnivorous mammals of the cat family.",
        "Penguins are flightless birds living mostly in the Southern Hemisphere.",
        "Elephants communicate using low-frequency rumbles.",
        "Blue whales are the largest animals known to have ever existed.",
        "Ants are social insects that form highly organized colonies."
    ]
    print("\n--- Processing Phase 1 Data (Animal Facts) ---")
    for i, text in enumerate(phase1_data):
        print(f"Input P1.{i+1}: \"{text}\"")
        with torch.no_grad():
            # The 'continuous_learning' parameter in model.forward() enables the adapter's processing
            # The 'update_memory' parameter enables memory updates for HierarchicalMemory & adapter buffer
            outputs = model(input_texts=[text], update_memory=True, continuous_learning=True)
        
        if continual_adapter_module:
            # Conceptual logging: Accessing buffer directly for demo.
            # In a real scenario, buffer management is internal to the adapter.
            print(f"  Processed. Adapter buffer entries: {len(continual_adapter_module.buffer)}")
            # Example: Show a snippet of what's in buffer if it's simple tensors/data
            # if continual_adapter_module.buffer:
            #     print(f"    Sample from buffer (first item shape): {continual_adapter_module.buffer[0].shape if isinstance(continual_adapter_module.buffer[0], torch.Tensor) else type(continual_adapter_module.buffer[0])}")
        if hierarchical_memory_module:
            # Conceptual: We can't easily inspect LTM/PM updates without modifying the class for logging.
            # We assume novelty detection and contextual gates are active.
            pass
    print("--- Phase 1 Data Processing Complete ---")
    print("-" * 40)

    # Phase 2 Data: Space Exploration
    phase2_data = [
        "Rockets use propellant to achieve liftoff and escape Earth's gravity.",
        "The International Space Station orbits our planet at high speed.",
        "Mars is known as the Red Planet due to iron oxide on its surface.",
        "Black holes are regions of spacetime where gravity is extremely strong.",
        "Telescopes allow us to observe distant galaxies and nebulae."
    ]
    print("\n--- Processing Phase 2 Data (Space Exploration) ---")
    for i, text in enumerate(phase2_data):
        print(f"Input P2.{i+1}: \"{text}\"")
        with torch.no_grad():
            outputs = model(input_texts=[text], update_memory=True, continuous_learning=True)

        if continual_adapter_module:
            print(f"  Processed. Adapter buffer entries: {len(continual_adapter_module.buffer)}")
    print("--- Phase 2 Data Processing Complete ---")
    print("-" * 40)

    # 4. Demonstrate Memory/Adaptation (Conceptual)
    print("\n--- Demonstrating Effects (Conceptual) ---")

    if continual_adapter_module:
        print(f"\nContinualAdapter's Replay Buffer State:")
        print(f"  Total items in buffer: {len(continual_adapter_module.buffer)}")
        print("  The buffer now contains representations from both 'Animal Facts' and 'Space Exploration' domains.")
        print("  During actual training, these buffered samples would be used by the adapter to mitigate catastrophic forgetting "
              "when learning new tasks or adapting to data distribution shifts.")
        # You could try to print some info about the buffer content if it's simple enough
        # For example, if buffer stores (hidden_state, label) tuples, you could show shapes.
        # Since it stores hidden_states directly:
        if len(continual_adapter_module.buffer) > 0:
            sample_buffered_item = continual_adapter_module.buffer[0]
            if isinstance(sample_buffered_item, torch.Tensor):
                 print(f"  Example buffered item (hidden state) shape: {sample_buffered_item.shape}")

    if hierarchical_memory_module:
        print(f"\nHierarchical Memory State:")
        print(f"  Short-Term Memory active slots: {hierarchical_memory_module.short_term_memory.memory_usage.sum().item() if hasattr(hierarchical_memory_module.short_term_memory, 'memory_usage') else 'N/A'}")
        print(f"  Long-Term Memory active slots: {hierarchical_memory_module.long_term_memory.memory_usage.sum().item() if hasattr(hierarchical_memory_module.long_term_memory, 'memory_usage') else 'N/A'}")
        print(f"  Persistent Memory active entries: {hierarchical_memory_module.persistent_memory.active_entries if hasattr(hierarchical_memory_module.persistent_memory, 'active_entries') else 'N/A'}")
        print("  HierarchicalMemory's novelty detection would have influenced what information was consolidated "
              "into LTM and PM from both domains. Contextual gates would help retrieve relevant information "
              "based on the current input's similarity to stored memories.")

    print("\nOverall Lifelong Learning Implication:")
    print("In a full application, the `ContinualAdapter` would help the model learn Phase 2 data "
          "more effectively while minimizing interference with Phase 1 knowledge. "
          "The `HierarchicalMemory` would store and organize information from both phases, "
          "allowing for contextually relevant retrieval and gradual knowledge consolidation.")
    print("This demo shows the components are active. True lifelong learning requires extensive training and evaluation.")
    print("-" * 40)
    print("Lifelong Learning Demonstration finished.")

if __name__ == "__main__":
    run_lifelong_learning_demo()
    # To run this:
    # 1. Ensure NeuroLiteModel correctly instantiates ContinualAdapter based on config.
    # 2. Ensure the 'continuous_learning' parameter in NeuroLiteModel.forward() correctly gates the adapter.
    # (These were addressed in previous subtasks).
    # Example: python examples/lifelong_learning_demo.py
