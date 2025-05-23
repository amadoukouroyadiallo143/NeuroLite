import pytest
import torch
from neurolite.model import NeuroLiteModel
from neurolite.config import NeuroLiteConfig
from neurolite.continual import ContinualAdapter

# Fixture for a basic NeuroLiteConfig
@pytest.fixture
def base_config():
    return NeuroLiteConfig.tiny() # Using tiny for speed

# Fixture for a NeuroLiteConfig with ContinualAdapter enabled
@pytest.fixture
def continual_config():
    config = NeuroLiteConfig.tiny()
    config.use_continual_adapter = True
    config.continual_adapter_buffer_size = 10
    config.continual_adapter_rate = 0.05
    config.continual_adapter_drift_threshold = 0.6
    return config

def test_neurolitemodel_instantiation_without_adapter(base_config):
    """Test basic model instantiation without any special modules."""
    model = NeuroLiteModel(config=base_config)
    assert model is not None
    assert model.continual_adapter is None
    assert model.symbolic is None # Default for tiny config
    assert model.bayesian_network is None # Default for tiny config

def test_neurolitemodel_instantiation_with_adapter(continual_config):
    """Test model instantiation with ContinualAdapter enabled."""
    model = NeuroLiteModel(config=continual_config)
    assert model is not None
    assert model.continual_adapter is not None
    assert isinstance(model.continual_adapter, ContinualAdapter)
    assert model.continual_adapter.buffer_size == continual_config.continual_adapter_buffer_size
    assert model.continual_adapter.adaptation_rate == continual_config.continual_adapter_rate
    assert model.continual_adapter.drift_threshold == continual_config.continual_adapter_drift_threshold

def test_neurolitemodel_forward_pass_with_adapter_no_update(continual_config):
    """Test forward pass with adapter enabled but not actively learning/updating buffer."""
    model = NeuroLiteModel(config=continual_config)
    model.eval() # Ensure deterministic behavior for dropout, etc.

    batch_size = 2
    seq_len = 5
    # Use input_ids as the TokenizedMinHashProjection is default for tiny if vocab_size is default
    # and input_projection_type is not explicitly set otherwise for text.
    # Or, we can use input_texts if we ensure MinHashBloomProjection is used.
    # For tiny config, input_projection_type defaults to "minhash_bloom" if vocab_size is not 0.
    # Let's use input_texts, as it's simpler for dummy data.
    # The default tiny() config has vocab_size=30000, so it will use TokenizedMinHashProjection.
    # We need to provide input_ids or change the projection type for text.
    # Easiest: use input_ids.
    dummy_input_ids = torch.randint(0, continual_config.vocab_size, (batch_size, seq_len))
    
    outputs = model(input_ids=dummy_input_ids, update_memory=False, continuous_learning=False)
    
    assert "hidden_states" in outputs
    assert outputs["hidden_states"].shape == (batch_size, seq_len, continual_config.hidden_size)
    if model.continual_adapter:
        assert len(model.continual_adapter.buffer) == 0 # Buffer should not be updated

def test_neurolitemodel_forward_pass_with_adapter_update_buffer(continual_config):
    """Test forward pass with adapter enabled and updating buffer."""
    model = NeuroLiteModel(config=continual_config)
    model.eval() 

    batch_size = 2
    seq_len = 5
    dummy_input_ids = torch.randint(0, continual_config.vocab_size, (batch_size, seq_len))

    # First pass: update buffer
    outputs1 = model(input_ids=dummy_input_ids, update_memory=True, continuous_learning=True)
    assert "hidden_states" in outputs1
    assert outputs1["hidden_states"].shape == (batch_size, seq_len, continual_config.hidden_size)
    
    expected_buffer_size_after_1_pass = batch_size # Adapter stores per-sample hidden states (mean over seq_len)
    if model.continual_adapter:
        assert len(model.continual_adapter.buffer) == expected_buffer_size_after_1_pass
        # Check content of buffer - it should be tensors of hidden_size
        if len(model.continual_adapter.buffer) > 0:
            assert model.continual_adapter.buffer[0].shape == (continual_config.hidden_size,)

    # Second pass with different data
    dummy_input_ids_2 = torch.randint(0, continual_config.vocab_size, (batch_size, seq_len))
    outputs2 = model(input_ids=dummy_input_ids_2, update_memory=True, continuous_learning=True)
    assert "hidden_states" in outputs2
    
    expected_buffer_size_after_2_passes = 2 * batch_size
    if model.continual_adapter:
        assert len(model.continual_adapter.buffer) == expected_buffer_size_after_2_passes

    # Test buffer limits
    num_passes_to_fill_buffer = (continual_config.continual_adapter_buffer_size // batch_size) + 1
    for i in range(num_passes_to_fill_buffer):
        current_input_ids = torch.randint(0, continual_config.vocab_size, (batch_size, seq_len))
        model(input_ids=current_input_ids, update_memory=True, continuous_learning=True)
    
    if model.continual_adapter:
        assert len(model.continual_adapter.buffer) == continual_config.continual_adapter_buffer_size


def test_neurolitemodel_forward_pass_adapter_disabled_config(base_config):
    """Test forward pass when adapter is disabled in config."""
    base_config.use_continual_adapter = False # Explicitly disable
    model = NeuroLiteModel(config=base_config)
    model.eval()

    batch_size = 2
    seq_len = 5
    dummy_input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_len))
    
    # Pass continuous_learning=True, but it should have no effect as adapter is None
    outputs = model(input_ids=dummy_input_ids, update_memory=True, continuous_learning=True)
    
    assert "hidden_states" in outputs
    assert model.continual_adapter is None

# Test with input_texts to ensure compatibility if projection changes
@pytest.fixture
def continual_config_text():
    config = NeuroLiteConfig.tiny()
    config.input_projection_type = "minhash_bloom" # Force MinHashBloomProjection for texts
    config.use_continual_adapter = True
    config.continual_adapter_buffer_size = 5
    return config

def test_neurolitemodel_forward_pass_with_adapter_text_input(continual_config_text):
    """Test forward pass with adapter and text inputs."""
    model = NeuroLiteModel(config=continual_config_text)
    model.eval()

    dummy_texts = ["hello world example one", "another example sentence here"]
    batch_size = len(dummy_texts)
    
    outputs = model(input_texts=dummy_texts, update_memory=True, continuous_learning=True)
    assert "hidden_states" in outputs
    # For MinHashBloom, seq_len becomes 1 per text.
    assert outputs["hidden_states"].shape == (batch_size, 1, continual_config_text.hidden_size)
    
    if model.continual_adapter:
        assert len(model.continual_adapter.buffer) == batch_size
        if len(model.continual_adapter.buffer) > 0:
            assert model.continual_adapter.buffer[0].shape == (continual_config_text.hidden_size,)

# Add more tests as needed, e.g., interaction with other modules if relevant.
# For now, these cover basic integration.
