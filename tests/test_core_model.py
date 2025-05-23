import torch
from neurolite.model import NeuroLiteModel, NeuroLiteConfig

def test_neurolite_model_forward_pass():
    """
    Tests the forward pass of NeuroLiteModel with a tiny configuration.
    """
    config = NeuroLiteConfig.tiny()
    model = NeuroLiteModel(config)

    # Dummy input texts
    texts = ["hello world", "this is a test"]
    
    # Process input (this might need adjustment based on actual model input processing)
    # Assuming model.process_input is a method that tokenizes and prepares input
    # For now, let's assume the model itself handles tokenization or expects tokenized input
    # If direct text input isn't supported, this test will need to be adapted
    # based on how text is converted to input IDs for the model.

    # For the purpose of this basic test, let's simulate tokenized input
    # based on typical transformer models.
    # Replace with actual tokenization if available.
    max_seq_length = config.max_position_embeddings
    # Dummy input_ids and attention_mask
    input_ids = torch.randint(0, config.vocab_size, (len(texts), max_seq_length))
    attention_mask = torch.ones_like(input_ids)

    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Assert output shape
    # The output is typically a tuple, with the first element being the hidden states
    last_hidden_states = outputs.last_hidden_state
    assert last_hidden_states.shape == (len(texts), max_seq_length, config.hidden_size), \
        f"Expected output shape {(len(texts), max_seq_length, config.hidden_size)}, but got {last_hidden_states.shape}"

def test_neurolite_model_tiny_config_instantiation():
    """Tests if NeuroLiteModel can be instantiated with a tiny config."""
    try:
        config = NeuroLiteConfig.tiny()
        model = NeuroLiteModel(config)
        assert model is not None, "Model instantiation failed."
    except Exception as e:
        assert False, f"Error during model instantiation with tiny config: {e}"
