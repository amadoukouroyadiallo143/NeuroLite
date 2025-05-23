import pytest
import torch
import json
import os
from neurolite.symbolic import NeuralSymbolicLayer
from neurolite.config import NeuroLiteConfig

# Fixture for a dummy rules.json file
@pytest.fixture(scope="module") # Use module scope for efficiency if file content is static
def dummy_rules_file(tmp_path_factory):
    rules_content = {
        "facts": ["is_a(cat, animal)", "type(E_global, global_type)"], # Added a global fact
        "rules": [
            {
                "premises": ["similaire(?x, ?y)", "is_a(?x, animal)"], # Assuming 'similaire' comes from neural extraction
                "conclusion": "related_animals(?x, ?y)"
            },
            {
                "premises": ["has_property(?x, furry)", "is_a(?x, animal)"],
                "conclusion": "is_pet(?x)"
            },
            { # Rule that might be triggered by dummy extraction
                "premises": ["similaire(B0_E0, B0_E1)"], # Specific to batch 0, entities 0 and 1
                "conclusion": "test_specific_rule_fired(B0_E0)"
            }
        ]
    }
    # Use tmp_path_factory for module-scoped fixture
    # For function scope, just tmp_path is fine.
    file_path = tmp_path_factory.mktemp("data") / "dummy_rules.json"
    with open(file_path, 'w') as f:
        json.dump(rules_content, f)
    return str(file_path)

@pytest.fixture
def symbolic_config(dummy_rules_file):
    config = NeuroLiteConfig.base_symbolic() # This already sets use_symbolic_module = True
    config.symbolic_rules_file = dummy_rules_file
    config.hidden_size = 128 # Keep it small for tests
    config.max_entities = 5 # For _extract_entities
    # For learnable embeddings in NeuralSymbolicLayer
    config.max_predicate_types = 10
    config.max_entities_in_vocab = 20 
    return config

def test_neural_symbolic_layer_instantiation(symbolic_config):
    try:
        layer = NeuralSymbolicLayer(
            hidden_size=symbolic_config.hidden_size,
            entity_extraction_threshold=0.1, # Lower threshold for dummy data
            max_entities=symbolic_config.max_entities,
            symbolic_rules_file=symbolic_config.symbolic_rules_file,
            dropout_rate=symbolic_config.dropout_rate,
            config=symbolic_config # Pass the whole config for new vocab params
        )
        assert layer is not None
        assert layer.rule_engine is not None
        # Check if global fact is loaded
        assert "is_a(cat, animal)" in layer.rule_engine.facts["is_a"]
        assert "type(E_global, global_type)" in layer.rule_engine.facts["type"]
    except Exception as e:
        pytest.fail(f"NeuralSymbolicLayer instantiation failed: {e}")

def test_neural_symbolic_layer_forward_pass_batch(symbolic_config):
    layer = NeuralSymbolicLayer(
        hidden_size=symbolic_config.hidden_size,
        entity_extraction_threshold=0.1,
        max_entities=symbolic_config.max_entities,
        symbolic_rules_file=symbolic_config.symbolic_rules_file,
        dropout_rate=symbolic_config.dropout_rate,
        config=symbolic_config
    )
    
    batch_size = 2
    seq_len = 10
    # Ensure hidden_states are not all zeros to avoid issues with norm/similarity calculations
    hidden_states = torch.randn(batch_size, seq_len, symbolic_config.hidden_size) + 1.0 

    # Set training to False to ensure symbolic module is tried (if not random)
    # Or ensure the random chance passes. For testing, better to control.
    # The layer's forward has: `if not self.training or torch.rand(1).item() > 0.5:`
    # To guarantee execution for test:
    layer.train(False) # Set to eval mode

    try:
        output_states = layer(hidden_states)
        assert output_states.shape == hidden_states.shape, \
            f"Expected output shape {hidden_states.shape}, but got {output_states.shape}"
        
        # Check if output is different from input (suggesting some processing happened)
        # This is a weak check as dropout alone would change it.
        # A stronger check would be if symbolic integration added non-zero values.
        assert not torch.allclose(output_states, hidden_states)

    except Exception as e:
        pytest.fail(f"NeuralSymbolicLayer forward pass failed: {e}")

def test_get_or_create_idx_logic(symbolic_config):
    layer = NeuralSymbolicLayer(
        hidden_size=symbolic_config.hidden_size,
        symbolic_rules_file=symbolic_config.symbolic_rules_file,
        config=symbolic_config
    )
    
    # Test for predicate vocab
    idx1, layer.next_predicate_idx = layer._get_or_create_idx("pred1", layer.predicate_to_idx, layer.next_predicate_idx, layer.max_predicate_types)
    assert idx1 == 0
    assert layer.next_predicate_idx == 1
    assert layer.predicate_to_idx["pred1"] == 0

    idx2, layer.next_predicate_idx = layer._get_or_create_idx("pred2", layer.predicate_to_idx, layer.next_predicate_idx, layer.max_predicate_types)
    assert idx2 == 1
    assert layer.next_predicate_idx == 2

    idx1_again, layer.next_predicate_idx = layer._get_or_create_idx("pred1", layer.predicate_to_idx, layer.next_predicate_idx, layer.max_predicate_types)
    assert idx1_again == 0
    assert layer.next_predicate_idx == 2 # Should not change

    # Test vocab limit
    # Fill up predicate_to_idx to max_predicate_types - 1 (since pred1 and pred2 are already in)
    current_max = layer.max_predicate_types
    for i in range(layer.next_predicate_idx, current_max):
        _, layer.next_predicate_idx = layer._get_or_create_idx(f"pred{i}", layer.predicate_to_idx, layer.next_predicate_idx, current_max)
    
    assert layer.next_predicate_idx == current_max
    
    # Try to add one more, should get back index 0 (our <UNK> placeholder)
    idx_unk, layer.next_predicate_idx = layer._get_or_create_idx("pred_overflow", layer.predicate_to_idx, layer.next_predicate_idx, current_max)
    assert idx_unk == 0 # Assuming 0 is UNK if vocab is full
    assert layer.next_predicate_idx == current_max # next_idx should not have incremented


# More focused test for symbolic integration if possible
@pytest.mark.skip(reason="Complex to guarantee specific rule firing with random dummy data, conceptual test")
def test_symbolic_integration_effect(symbolic_config):
    layer = NeuralSymbolicLayer(
        hidden_size=symbolic_config.hidden_size,
        entity_extraction_threshold=0.01, # Very low to extract something
        max_entities=2, # Force a few entities
        symbolic_rules_file=symbolic_config.symbolic_rules_file,
        dropout_rate=0.0, # Disable dropout for this test
        config=symbolic_config
    )
    layer.train(False) # Eval mode

    batch_size = 1 # Simpler to analyze for one item
    seq_len = 5
    hidden_states = torch.rand(batch_size, seq_len, symbolic_config.hidden_size) * 2 - 1 # [-1, 1]

    # To make "similaire(B0_E0, B0_E1)" potentially fire, we need _extract_entities
    # to produce at least two entities for batch item 0, and _extract_relations
    # to generate that specific predicate.
    # The dummy _extract_relations creates "similaire" if cosine similarity > 0.7.
    # We can try to craft input hidden_states such that entity_extractor output is similar.
    
    # This is very hard to control perfectly.
    # A simpler check: if any rule fires, the output of _integrate_symbolic_results
    # before projection should be non-zero.

    # Pass 1: Normal run
    output_states_with_symbolic = layer(hidden_states)

    # Pass 2: Run with an empty rules file (or no rules that can fire)
    empty_rules_path = symbolic_config.symbolic_rules_file.replace("dummy_rules.json", "empty_rules_for_test.json")
    with open(empty_rules_path, 'w') as f:
        json.dump({"facts": [], "rules": []}, f)
    
    layer_no_rules = NeuralSymbolicLayer(
        hidden_size=symbolic_config.hidden_size,
        entity_extraction_threshold=0.01,
        max_entities=2,
        symbolic_rules_file=empty_rules_path, # Use empty rules
        dropout_rate=0.0,
        config=symbolic_config
    )
    layer_no_rules.train(False)
    output_states_no_symbolic = layer_no_rules(hidden_states)

    # If symbolic processing added something, the outputs should differ more than just noise.
    # This assumes that the dummy rule "test_specific_rule_fired" was actually fired.
    # This test remains conceptual due to the difficulty of guaranteeing rule firing.
    
    # A more direct way would be to mock _process_symbolic to return specific derived facts.
    # For now, we assert that if rules are present and processing happens, output might change.
    # This is not a strong assertion for specific symbolic logic.
    assert not torch.allclose(output_states_with_symbolic, output_states_no_symbolic, atol=1e-5), \
        "Output with symbolic rules should ideally differ from output with no rules if rules fired."

    if os.path.exists(empty_rules_path):
        os.remove(empty_rules_path)

# Test specific parts of the batch processing logic if they can be isolated.
# For example, testing _extract_entities with a known input.
def test_extract_entities_batch_output_format(symbolic_config):
    layer = NeuralSymbolicLayer(
        hidden_size=symbolic_config.hidden_size,
        entity_extraction_threshold=0.1,
        max_entities=symbolic_config.max_entities,
        symbolic_rules_file=symbolic_config.symbolic_rules_file,
        config=symbolic_config
    )
    batch_size = 3
    seq_len = 7
    hidden_states = torch.randn(batch_size, seq_len, symbolic_config.hidden_size)
    
    batch_entity_ids, batch_entity_embeds = layer._extract_entities(hidden_states)
    
    assert isinstance(batch_entity_ids, list), "batch_entity_ids should be a list"
    assert len(batch_entity_ids) == batch_size, f"Expected {batch_size} lists of entity_ids"
    assert isinstance(batch_entity_embeds, list), "batch_entity_embeds should be a list"
    assert len(batch_entity_embeds) == batch_size, f"Expected {batch_size} lists of entity_embeds"

    for i in range(batch_size):
        assert isinstance(batch_entity_ids[i], list), f"Item {i} in batch_entity_ids should be a list"
        # batch_entity_embeds[i] should be a tensor, or a list of tensors if not stacked.
        # Current implementation stacks them, so it's a tensor.
        assert isinstance(batch_entity_embeds[i], torch.Tensor), f"Item {i} in batch_entity_embeds should be a tensor"
        
        num_entities_item_i = len(batch_entity_ids[i])
        if num_entities_item_i > 0:
            assert batch_entity_embeds[i].shape[0] == num_entities_item_i
            assert batch_entity_embeds[i].shape[1] == symbolic_config.hidden_size // 2
            for entity_id_str in batch_entity_ids[i]:
                assert isinstance(entity_id_str, str)
                assert entity_id_str.startswith(f"B{i}_E")
        else:
            assert batch_entity_embeds[i].numel() == 0 # Empty tensor if no entities

# It's important to ensure the test environment can find the 'neurolite' package.
# This usually means running pytest from the project root or having the package installed.
# If "No space left on device" was an issue for installing dependencies,
# these tests might fail at import if torch is not available.
# The tests themselves don't install anything, but rely on the environment.
