import torch
from neurolite.mixer import MLPMixerLayer

def test_mlp_mixer_layer_forward_pass():
    """
    Tests the forward pass of MLPMixerLayer.
    """
    batch_size = 2
    seq_len = 10
    hidden_size = 128
    token_mixing_hidden_size = 256
    channel_mixing_hidden_size = 512

    # Instantiate the layer
    mixer_layer = MLPMixerLayer(
        hidden_size=hidden_size,
        token_mixing_hidden_size=token_mixing_hidden_size,
        channel_mixing_hidden_size=channel_mixing_hidden_size
    )

    # Dummy input tensor
    dummy_input = torch.randn(batch_size, seq_len, hidden_size)

    # Forward pass
    output = mixer_layer(dummy_input)

    # Assert output shape
    assert output.shape == dummy_input.shape, \
        f"Expected output shape {dummy_input.shape}, but got {output.shape}"

def test_mlp_mixer_layer_instantiation():
    """Tests if MLPMixerLayer can be instantiated."""
    try:
        mixer_layer = MLPMixerLayer(
            hidden_size=128,
            token_mixing_hidden_size=256,
            channel_mixing_hidden_size=512
        )
        assert mixer_layer is not None, "MLPMixerLayer instantiation failed."
    except Exception as e:
        assert False, f"Error during MLPMixerLayer instantiation: {e}"
