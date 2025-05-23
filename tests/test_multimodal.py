import pytest
import torch
from neurolite.multimodal import MultimodalProjection, CrossModalAttention
from neurolite.config import NeuroLiteConfig # For default values

# Fixture for MultimodalProjection configuration parameters
@pytest.fixture
def mpp_config_params():
    # Using values similar to what NeuroLiteConfig.tiny() might imply for hidden_size,
    # and some defaults for multimodal specific parts.
    # NeuroLiteConfig.tiny() has hidden_size=128.
    # MultimodalProjection by default sets output_dim to hidden_size.
    return {
        "output_dim": 128, # Corresponds to hidden_size in NeuroLiteConfig
        "minhash_permutations": 64, # Smaller for tests
        "bloom_filter_size": 256,   # Smaller for tests
        "image_patch_size": 16,     # Standard
        "video_num_sampled_frames": 3,
        "dropout_rate": 0.0 # Disable dropout for predictable tests
    }

# --- Tests for MultimodalProjection ---

def test_multimodalprojection_instantiation(mpp_config_params):
    try:
        mpp = MultimodalProjection(**mpp_config_params)
        assert mpp is not None
        assert mpp.output_dim == mpp_config_params["output_dim"]
        assert mpp.video_num_sampled_frames == mpp_config_params["video_num_sampled_frames"]
        assert mpp.text_encoder is not None
        assert mpp.image_encoder is not None
        assert mpp.audio_encoder is not None
        assert mpp.video_frame_processor is not None # Aliased to image_encoder
    except Exception as e:
        pytest.fail(f"MultimodalProjection instantiation failed: {e}")

@pytest.mark.parametrize("modality_key, input_data_fn", [
    ("text", lambda: ["sample text one for test.", "another sample document."]),
    ("image", lambda: torch.randn(2, 3, 224, 224)), # B, C, H, W - H,W for 14x14 patches with patch_size=16
    ("audio", lambda: torch.randn(2, 1, 128, 80)), # B, C, T, F - T,F for audio_encoder
    ("video", lambda: torch.randn(2, 5, 3, 224, 224)) # B, Frames, C, H, W
])
def test_mpp_forward_individual_modalities(mpp_config_params, modality_key, input_data_fn):
    mpp = MultimodalProjection(**mpp_config_params)
    mpp.eval() # For consistent behavior (dropout is already 0.0)
    
    batch_size = 2 
    dummy_input = input_data_fn()
    
    # Adjust batch size for text input if lambda doesn't match
    if modality_key == "text" and isinstance(dummy_input, list):
        if len(dummy_input) != batch_size:
            dummy_input = dummy_input * (batch_size // len(dummy_input) + 1)
            dummy_input = dummy_input[:batch_size]
    
    inputs = {modality_key: dummy_input}
    
    with torch.no_grad():
        output = mpp(inputs)
        
    assert output.shape == (batch_size, mpp_config_params["output_dim"]), \
        f"Output shape mismatch for modality {modality_key}"
    assert not torch.all(output == 0), f"Output for modality {modality_key} is all zeros."

def test_mpp_forward_all_modalities(mpp_config_params):
    mpp = MultimodalProjection(**mpp_config_params)
    mpp.eval()
    batch_size = 2
    
    inputs = {
        "text": ["text one", "text two"][:batch_size],
        "image": torch.randn(batch_size, 3, 224, 224),
        "audio": torch.randn(batch_size, 1, 128, 80),
        "video": torch.randn(batch_size, mpp_config_params["video_num_sampled_frames"] + 2, 3, 224, 224)
    }
    
    with torch.no_grad():
        output = mpp(inputs)
        
    assert output.shape == (batch_size, mpp_config_params["output_dim"])
    assert not torch.all(output == 0)

def test_mpp_return_individual_modalities(mpp_config_params):
    mpp = MultimodalProjection(**mpp_config_params)
    mpp.eval()
    batch_size = 2
    output_dim = mpp_config_params["output_dim"]

    inputs = {
        "text": ["sample text", "another sample"][:batch_size],
        "image": torch.randn(batch_size, 3, 224, 224)
        # Audio and video are omitted for this test
    }
    
    with torch.no_grad():
        fused_repr, individual_reprs = mpp(inputs, return_individual_modalities=True)
        
    assert isinstance(fused_repr, torch.Tensor)
    assert fused_repr.shape == (batch_size, output_dim)
    assert isinstance(individual_reprs, dict)
    
    assert "text" in individual_reprs
    assert individual_reprs["text"].shape == (batch_size, output_dim)
    assert not torch.all(individual_reprs["text"] == 0)
    
    assert "image" in individual_reprs
    assert individual_reprs["image"].shape == (batch_size, output_dim)
    assert not torch.all(individual_reprs["image"] == 0)
    
    assert "audio" not in individual_reprs # Since it wasn't provided and default zero tensor shouldn't be added
    assert "video" not in individual_reprs

def test_mpp_video_frame_sampling(mpp_config_params):
    mpp = MultimodalProjection(**mpp_config_params)
    mpp.eval()
    batch_size = 1
    num_sampled_frames = mpp_config_params["video_num_sampled_frames"]

    # Case 1: Fewer frames than num_sampled_frames
    video_few_frames = torch.randn(batch_size, num_sampled_frames - 1, 3, 224, 224)
    inputs_few = {"video": video_few_frames}
    with torch.no_grad():
        output_few, indiv_few = mpp(inputs_few, return_individual_modalities=True)
    assert output_few.shape == (batch_size, mpp_config_params["output_dim"])
    assert "video" in indiv_few
    assert indiv_few["video"].shape == (batch_size, mpp_config_params["output_dim"])
    # The internal `sampled_frames` would have num_sampled_frames-1 frames.

    # Case 2: More frames than num_sampled_frames
    video_more_frames = torch.randn(batch_size, num_sampled_frames + 5, 3, 224, 224)
    inputs_more = {"video": video_more_frames}
    with torch.no_grad():
        output_more, indiv_more = mpp(inputs_more, return_individual_modalities=True)
    assert output_more.shape == (batch_size, mpp_config_params["output_dim"])
    assert "video" in indiv_more
    assert indiv_more["video"].shape == (batch_size, mpp_config_params["output_dim"])
    # Internally, `sampled_frames` should have `num_sampled_frames` frames.

    # Case 3: Zero frames in input video tensor
    video_zero_frames = torch.randn(batch_size, 0, 3, 224, 224)
    inputs_zero = {"video": video_zero_frames}
    with torch.no_grad():
        output_zero, indiv_zero = mpp(inputs_zero, return_individual_modalities=True)
    assert output_zero.shape == (batch_size, mpp_config_params["output_dim"])
    assert "video" not in indiv_zero # video_repr should remain zeros and not added
    assert torch.all(output_zero == 0) # Since only video was input and it was empty


# --- Tests for CrossModalAttention ---

@pytest.fixture
def cma_config_params():
    return {
        "hidden_size": 128,
        "num_heads": 4,
        "dropout_rate": 0.0 # Disable dropout
    }

def test_crossmodalattention_instantiation(cma_config_params):
    try:
        cma = CrossModalAttention(**cma_config_params)
        assert cma is not None
        assert cma.num_heads == cma_config_params["num_heads"]
        assert cma.head_size == cma_config_params["hidden_size"] // cma_config_params["num_heads"]
    except Exception as e:
        pytest.fail(f"CrossModalAttention instantiation failed: {e}")

def test_cma_forward_pass(cma_config_params):
    cma = CrossModalAttention(**cma_config_params)
    cma.eval()

    batch_size = 2
    seq_len_q = 10 # Query sequence length
    seq_len_kv = 5  # Key/Value sequence length
    hidden_size = cma_config_params["hidden_size"]

    dummy_query = torch.randn(batch_size, seq_len_q, hidden_size)
    dummy_kv = torch.randn(batch_size, seq_len_kv, hidden_size)

    with torch.no_grad():
        output = cma(query_modality=dummy_query, key_value_modality=dummy_kv)
        
    assert output.shape == (batch_size, seq_len_q, hidden_size), \
        f"Output shape mismatch. Expected {(batch_size, seq_len_q, hidden_size)}, got {output.shape}"
    assert not torch.allclose(output, dummy_query), "Output should be different from query if attention applied."

def test_cma_forward_pass_with_mask(cma_config_params):
    cma = CrossModalAttention(**cma_config_params)
    cma.eval()

    batch_size = 1 # Simpler for mask inspection
    seq_len_q = 3
    seq_len_kv = 4
    hidden_size = cma_config_params["hidden_size"]

    dummy_query = torch.randn(batch_size, seq_len_q, hidden_size)
    dummy_kv = torch.randn(batch_size, seq_len_kv, hidden_size)
    
    # Mask: query token 0 can only attend to kv token 0 and 2.
    # query token 1 can attend to all kv tokens.
    # query token 2 can only attend to kv token 3.
    # Mask shape [B, SeqLenQ, SeqLenKV]
    attention_mask = torch.tensor([[[1, 0, 1, 0],
                                    [1, 1, 1, 1],
                                    [0, 0, 0, 1]]], dtype=torch.bool)
    
    with torch.no_grad():
        output = cma(query_modality=dummy_query, key_value_modality=dummy_kv, attention_mask=attention_mask)
        
    assert output.shape == (batch_size, seq_len_q, hidden_size)
    # Further checks could involve inspecting attention weights if the layer returned them,
    # but that's beyond the scope of this basic forward pass test.
    # The main thing is that it runs without error with a mask.
