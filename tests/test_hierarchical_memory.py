import pytest
import torch
from neurolite.hierarchical_memory import HierarchicalMemory
from neurolite.config import NeuroLiteConfig # Assuming this is needed for config defaults

# Fixture for a basic HierarchicalMemory configuration
@pytest.fixture
def HMemConfig():
    config = NeuroLiteConfig.tiny() # Use tiny as a base for hidden_size etc.
    # HierarchicalMemory specific defaults are mostly fine, but we can override
    config.hidden_size = 64 # Keep small for tests
    config.short_term_size = 16
    config.long_term_size = 32
    config.persistent_size = 64
    config.novelty_threshold_ltm = 0.7 # High threshold for testing "no update"
    config.novelty_threshold_pm = 0.7  # High threshold for testing "no update"
    return config

# Helper function to access internal memory_weights for testing Task 2
# This involves temporarily modifying the class or making _process_chunk return more info.
# For this test, we'll assume _process_chunk can be made to return memory_weights for inspection.
# This is a common pattern: add a temporary debug flag or return for testability.

class TestableHierarchicalMemory(HierarchicalMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_memory_weights_for_test = None
        self.last_update_ltm_flag_for_test = None
        self.last_update_pm_flag_for_test = None

    def _process_chunk(
        self, 
        hidden_states: torch.Tensor,
        update_memory: bool = True,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # --- Existing _process_chunk logic up to memory_weights calculation ---
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        current_time = torch.tensor(0.0, device=device) # Simplified time for testing
        self.last_access_time = current_time
        forgetting_factor = 1.0 # Disable forgetting for stable tests unless specifically testing it

        queries = self.query_projection(hidden_states)
        update_mem_base_flag = update_memory
        
        update_ltm_flag = False
        update_pm_flag = False

        if update_mem_base_flag:
            if self.long_term_memory.memory_keys.numel() > 0:
                avg_queries_for_ltm_novelty = queries.mean(dim=1, keepdim=True)
                novelty_for_ltm = self._calculate_novelty_score(avg_queries_for_ltm_novelty, self.long_term_memory.memory_keys)
                if novelty_for_ltm > self.novelty_threshold_ltm:
                    update_ltm_flag = True
            else:
                update_ltm_flag = True

            if self.persistent_memory.memory_keys.numel() > 0 and self.persistent_memory.active_entries > 0:
                avg_queries_for_pm_novelty = queries.mean(dim=1, keepdim=True)
                novelty_for_pm = self._calculate_novelty_score(avg_queries_for_pm_novelty, self.persistent_memory.memory_keys[:self.persistent_memory.active_entries])
                if novelty_for_pm > self.novelty_threshold_pm:
                    update_pm_flag = True
            else:
                update_pm_flag = True
        
        self.last_update_ltm_flag_for_test = update_ltm_flag # Store for test
        self.last_update_pm_flag_for_test = update_pm_flag   # Store for test

        short_term_output = self.short_term_memory(queries, update_memory=update_mem_base_flag)
        long_term_output = self.long_term_memory(queries, update_memory=update_ltm_flag)
        persistent_output = self.persistent_memory(queries, update_memory=update_pm_flag)
        
        if update_memory and forgetting_factor < 1.0:
             with torch.no_grad():
                if hasattr(self.long_term_memory, 'memory_values'):
                    self.long_term_memory.memory_values *= forgetting_factor
                if hasattr(self.persistent_memory, 'memory_values'):
                    self.persistent_memory.memory_values *= forgetting_factor
        
        memory_weights = self.memory_gate(queries)
        self.last_memory_weights_for_test = memory_weights # Store for test inspection
        
        combined_memory = (
            memory_weights[..., 0:1] * short_term_output +
            memory_weights[..., 1:2] * long_term_output +
            memory_weights[..., 2:3] * persistent_output
        )
        
        if seq_len > 512 and not self.training: # Simplified condition from original
            attn_output = self._efficient_attention(hidden_states, combined_memory, attention_mask)
        else:
            attn_output, _ = self.memory_attention(
                hidden_states, combined_memory, combined_memory,
                key_padding_mask=~attention_mask if attention_mask is not None else None
            )
        
        output = self.output_projection(attn_output)
        return hidden_states + output


def test_hierarchicalmemory_instantiation(HMemConfig):
    memory = HierarchicalMemory(
        hidden_size=HMemConfig.hidden_size,
        short_term_size=HMemConfig.short_term_size,
        long_term_size=HMemConfig.long_term_size,
        persistent_size=HMemConfig.persistent_size,
        config=HMemConfig # Pass the config object
    )
    assert memory is not None
    assert memory.short_term_memory.memory_size == HMemConfig.short_term_size
    assert memory.long_term_memory.memory_size == HMemConfig.long_term_size
    assert memory.persistent_memory.memory_size == HMemConfig.persistent_size
    assert memory.novelty_threshold_ltm == HMemConfig.novelty_threshold_ltm

def test_contextual_memory_gates(HMemConfig):
    """Test if memory_weights have the correct per-token shape."""
    memory = TestableHierarchicalMemory( # Use the testable version
        hidden_size=HMemConfig.hidden_size,
        short_term_size=HMemConfig.short_term_size,
        config=HMemConfig
    )
    memory.eval() # Consistent behavior

    batch_size = 2
    seq_len = 10
    dummy_hidden_states = torch.randn(batch_size, seq_len, HMemConfig.hidden_size)

    _ = memory._process_chunk(dummy_hidden_states, update_memory=True)
    
    assert memory.last_memory_weights_for_test is not None
    assert memory.last_memory_weights_for_test.shape == (batch_size, seq_len, 3), \
        "Memory weights should be [batch_size, seq_len, 3] for per-token gating."

def test_novelty_based_consolidation_ltm(HMemConfig):
    """Test if LTM update is skipped for non-novel data and occurs for novel data."""
    # High novelty threshold to make it easier to trigger "not novel"
    HMemConfig.novelty_threshold_ltm = 0.8 
    memory = TestableHierarchicalMemory(
        hidden_size=HMemConfig.hidden_size,
        config=HMemConfig
    )
    memory.eval()

    batch_size = 1 
    seq_len = 5
    # Initial data to populate LTM
    hidden_states_A = torch.randn(batch_size, seq_len, HMemConfig.hidden_size)
    # Ensure LTM keys are populated for novelty calculation by processing once
    _ = memory._process_chunk(hidden_states_A, update_memory=True)
    # The first update should happen as LTM is empty or content is new
    assert memory.last_update_ltm_flag_for_test is True, "LTM should update with initial data"
    
    # Store current state of LTM (e.g., sum of values, or a specific slot if deterministic)
    # DifferentiableMemory updates its memory_values and memory_keys in place.
    # We can check if self.long_term_memory.memory_values changed.
    ltm_values_before_B = memory.long_term_memory.memory_values.clone()

    # Data very similar to A (should be non-novel)
    hidden_states_B = hidden_states_A + 1e-6 * torch.randn_like(hidden_states_A)
    _ = memory._process_chunk(hidden_states_B, update_memory=True)
    
    # Expect LTM update to be skipped due to low novelty
    assert memory.last_update_ltm_flag_for_test is False, \
        f"LTM update should be skipped for non-novel data. Novelty threshold: {memory.novelty_threshold_ltm}"
    assert torch.allclose(ltm_values_before_B, memory.long_term_memory.memory_values), \
        "LTM values should not change significantly for non-novel input when update is skipped."

    # Data very different from A (should be novel)
    hidden_states_C = torch.randn(batch_size, seq_len, HMemConfig.hidden_size) * 5 
    _ = memory._process_chunk(hidden_states_C, update_memory=True)

    # Expect LTM update to occur due to high novelty
    assert memory.last_update_ltm_flag_for_test is True, \
        "LTM update should occur for novel data."
    assert not torch.allclose(ltm_values_before_B, memory.long_term_memory.memory_values, atol=1e-5), \
        "LTM values should change after processing novel input C."


def test_novelty_based_consolidation_pm(HMemConfig):
    """Test if PM update is skipped for non-novel data and occurs for novel data."""
    HMemConfig.novelty_threshold_pm = 0.8
    memory = TestableHierarchicalMemory(
        hidden_size=HMemConfig.hidden_size,
        config=HMemConfig
    )
    memory.eval()

    batch_size = 1
    seq_len = 5
    hidden_states_A = torch.randn(batch_size, seq_len, HMemConfig.hidden_size)
    # Initial processing to ensure PM might get some entries.
    _ = memory._process_chunk(hidden_states_A, update_memory=True)
    # The first update should happen for PM if novelty_for_pm > threshold or PM is empty
    # This depends on whether LTM also updated and if PM's novelty relative to queries is high.
    # For this test, let's assume it updated.
    assert memory.last_update_pm_flag_for_test is True, "PM should attempt update with initial data"
    pm_values_before_B = memory.persistent_memory.memory_values.clone()
    pm_usage_before_B = memory.persistent_memory.memory_usage.clone()

    # Data very similar to A
    hidden_states_B = hidden_states_A + 1e-6 * torch.randn_like(hidden_states_A)
    _ = memory._process_chunk(hidden_states_B, update_memory=True)
    
    assert memory.last_update_pm_flag_for_test is False, \
        f"PM update should be skipped for non-novel data. Novelty threshold: {memory.novelty_threshold_pm}"
    assert torch.allclose(pm_values_before_B, memory.persistent_memory.memory_values), \
        "PM values should not change significantly for non-novel input if update is skipped."
    assert torch.allclose(pm_usage_before_B, memory.persistent_memory.memory_usage), \
        "PM usage should not change for non-novel input if update is skipped."


    # Data very different from A
    hidden_states_C = torch.randn(batch_size, seq_len, HMemConfig.hidden_size) * 5
    _ = memory._process_chunk(hidden_states_C, update_memory=True)

    assert memory.last_update_pm_flag_for_test is True, \
        "PM update should occur for novel data."
    # Check if PM changed. It's harder to guarantee a change if all slots are full and
    # the new data replaces an old one with similar usage.
    # A simpler check is if *any* value changed or if usage count changed (if new slot used).
    changed = not torch.allclose(pm_values_before_B, memory.persistent_memory.memory_values, atol=1e-5) or \
              not torch.allclose(pm_usage_before_B, memory.persistent_memory.memory_usage)
    assert changed, "PM values or usage should change after processing novel input C."


def test_calculate_novelty_score(HMemConfig):
    """Test the _calculate_novelty_score helper directly."""
    memory = HierarchicalMemory(hidden_size=HMemConfig.hidden_size, config=HMemConfig)
    
    key_dim = HMemConfig.hidden_size # Assuming key_size is hidden_size
    
    # Case 1: Empty memory bank
    input_keys = torch.randn(1, 5, key_dim)
    mem_keys_empty = torch.empty(0, key_dim)
    # Manually set active_entries to 0 for persistent_memory for this specific test path
    memory.persistent_memory.active_entries = 0 
    novelty = memory._calculate_novelty_score(input_keys, mem_keys_empty)
    assert novelty == 1.0, "Novelty should be 1.0 if memory is empty"

    # Case 2: Identical keys (low novelty)
    mem_keys_A = torch.randn(10, key_dim)
    memory.persistent_memory.memory_keys = mem_keys_A # Directly set for test
    memory.persistent_memory.active_entries = 10
    input_keys_same = mem_keys_A[0:1, :].unsqueeze(0) # Take one key from memory as input [1,1,key_dim]
    novelty_same = memory._calculate_novelty_score(input_keys_same, mem_keys_A)
    assert novelty_same < 0.01, f"Novelty should be close to 0 for identical keys, got {novelty_same}"

    # Case 3: Orthogonal keys (high novelty)
    # Create orthogonal keys (more complex, for simplicity use very different random keys)
    mem_keys_B = torch.randn(10, key_dim)
    memory.persistent_memory.memory_keys = mem_keys_B
    memory.persistent_memory.active_entries = 10
    input_keys_diff = torch.randn(1, 5, key_dim) * 10 # Make them statistically different
    
    # Ensure no accidental high similarity with mem_keys_B
    # This can happen by chance with random numbers. A true orthogonal set would be better.
    # For a robust test, one might construct input_keys_diff to be orthogonal to mem_keys_B.
    # For now, rely on statistical difference.
    
    novelty_diff = memory._calculate_novelty_score(input_keys_diff, mem_keys_B)
    assert novelty_diff > 0.5, f"Novelty should be high for very different keys, got {novelty_diff}"
    # The exact value depends on random initialization, so 0.5 is a loose lower bound.

# Note: The `TestableHierarchicalMemory` is a common technique for testing internals.
# In a production setting, you might use conditional debug flags or other methods
# if you need to inspect such internal states without subclassing.
