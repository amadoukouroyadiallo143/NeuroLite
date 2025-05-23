import pytest
import torch
from neurolite.symbolic import BayesianBeliefNetwork
from neurolite.config import NeuroLiteConfig # Assuming this is the correct import path

@pytest.fixture
def bayesian_config_random():
    config = NeuroLiteConfig() # Basic config
    config.hidden_size = 64 # Smaller for tests
    config.num_bayesian_variables = 5
    config.max_parents_bayesian = 2 # Used by random initializer
    config.bayesian_network_structure = None # Ensure random initialization
    return config

@pytest.fixture
def bayesian_config_predefined_chain():
    config = NeuroLiteConfig()
    config.hidden_size = 64
    config.num_bayesian_variables = 3
    # Structure: 0 -> 1 -> 2
    config.bayesian_network_structure = [(0, 1), (1, 2)]
    return config

@pytest.fixture
def bayesian_config_predefined_common_parent():
    config = NeuroLiteConfig()
    config.hidden_size = 64
    config.num_bayesian_variables = 3
    # Structure: 0 -> 1, 0 -> 2 (0 is common parent)
    config.bayesian_network_structure = [(0, 1), (0, 2)]
    return config


def test_bayesian_network_instantiation_random(bayesian_config_random):
    try:
        network = BayesianBeliefNetwork(config=bayesian_config_random, hidden_size=bayesian_config_random.hidden_size)
        assert network is not None
        assert network.num_variables == bayesian_config_random.num_bayesian_variables
        assert network.parents_matrix.shape == (network.num_variables, network.num_variables)
        # Check if topological order is valid
        assert len(network.topological_order) == network.num_variables
        assert sorted(network.topological_order) == list(range(network.num_variables))
    except Exception as e:
        pytest.fail(f"Instantiation with random structure failed: {e}")

def test_bayesian_network_instantiation_predefined(bayesian_config_predefined_chain):
    try:
        network = BayesianBeliefNetwork(config=bayesian_config_predefined_chain, hidden_size=bayesian_config_predefined_chain.hidden_size)
        assert network is not None
        assert network.num_variables == bayesian_config_predefined_chain.num_bayesian_variables
        # Check structure: 0 -> 1, 1 -> 2
        assert network.parents_matrix[1, 0].item() is True # Var 1 has parent 0
        assert network.parents_matrix[2, 1].item() is True # Var 2 has parent 1
        assert network.parents_matrix[0, :].sum() == 0 # Var 0 has no parents
        assert network.parents_matrix[1, :].sum() == 1
        assert network.parents_matrix[2, :].sum() == 1
        # Check children matrix for completeness
        assert network.children_matrix[0,1].item() is True
        assert network.children_matrix[1,2].item() is True

        # Check topological order for 0->1->2 should be [0, 1, 2]
        assert network.topological_order == [0, 1, 2]

    except Exception as e:
        pytest.fail(f"Instantiation with predefined structure failed: {e}")

def test_bayesian_network_forward_pass(bayesian_config_random):
    network = BayesianBeliefNetwork(config=bayesian_config_random, hidden_size=bayesian_config_random.hidden_size)
    batch_size = 2
    seq_len = 10
    dummy_hidden_states = torch.randn(batch_size, seq_len, bayesian_config_random.hidden_size)
    
    try:
        output_states = network(dummy_hidden_states)
        assert output_states.shape == dummy_hidden_states.shape, \
            f"Expected output shape {dummy_hidden_states.shape}, but got {output_states.shape}"
    except Exception as e:
        pytest.fail(f"Forward pass failed: {e}")

def test_bayesian_inference_conceptual_chain(bayesian_config_predefined_chain):
    network = BayesianBeliefNetwork(config=bayesian_config_predefined_chain, hidden_size=bayesian_config_predefined_chain.hidden_size)
    batch_size = 1 # Test with one batch item for simplicity
    
    # Evidence: Var 0 is True (high probability)
    # Structure: 0 -> 1 -> 2
    evidence = torch.full((batch_size, network.num_variables), 0.5) # Default to uncertain
    evidence[:, 0] = 0.9 # Var 0 is likely true
    
    # Call _infer_probabilities directly
    # Using a small number of samples for test speed, might not be very accurate
    posterior_probs = network._infer_probabilities(evidence, num_samples=200) 
    
    assert posterior_probs.shape == (batch_size, network.num_variables)
    
    # Conceptual checks for 0 -> 1 -> 2 chain with evidence P(Var0=1) = 0.9:
    # 1. P(Var0) should be close to 1.0 (as it's evidence)
    assert torch.isclose(posterior_probs[:, 0], torch.tensor(1.0), atol=0.1), \
        f"P(Var0) expected ~1.0, got {posterior_probs[:, 0].item()}"

    # 2. P(Var1) should be higher than if P(Var0) was low.
    #    This requires comparing to another run or making assumptions about CPTs.
    #    For now, just check it's a valid probability.
    assert 0.0 <= posterior_probs[:, 1].item() <= 1.0
    
    # 3. P(Var2) depends on P(Var1).
    assert 0.0 <= posterior_probs[:, 2].item() <= 1.0

    # More specific check: if P(0=True) is high, P(1=True|0=True) should be > P(1=True|0=False) (on average)
    # This is hard to test without knowing the CPT embeddings.
    # We can check if P(1) > some baseline (e.g. 0.5) if the CPTs are not totally random against it.
    # Given the similarity logic, if Var0's embedding for state 1 is somewhat aligned with Var1's CPT,
    # then P(Var1) should increase.
    
    # A simple check: If Var0=1, then Var1's probability should be influenced.
    # Let's run with Var0=0.1 (likely false)
    evidence_v0_false = torch.full((batch_size, network.num_variables), 0.5)
    evidence_v0_false[:, 0] = 0.1
    posterior_probs_v0_false = network._infer_probabilities(evidence_v0_false, num_samples=200)
    
    # We expect P(Var1 | Var0=0.9) to be different from P(Var1 | Var0=0.1)
    # This is not a guarantee of correctness but shows sensitivity.
    # The direction of change depends on the random CPT embeddings.
    if not torch.isclose(posterior_probs[:, 1], posterior_probs_v0_false[:, 1], atol=0.1):
         print(f"P(Var1|Var0=0.9) = {posterior_probs[:, 1].item()}, P(Var1|Var0=0.1) = {posterior_probs_v0_false[:, 1].item()}")
    # This is not a strict assertion because the random CPTs might coincidentally lead to similar outcomes.

def test_bayesian_inference_common_parent(bayesian_config_predefined_common_parent):
    # Structure: 0 -> 1, 0 -> 2
    network = BayesianBeliefNetwork(config=bayesian_config_predefined_common_parent, hidden_size=bayesian_config_predefined_common_parent.hidden_size)
    batch_size = 1
    
    evidence = torch.full((batch_size, network.num_variables), 0.5)
    evidence[:, 0] = 0.9 # Var 0 is likely True
    
    posterior_probs = network._infer_probabilities(evidence, num_samples=200)
    
    assert posterior_probs.shape == (batch_size, network.num_variables)
    assert torch.isclose(posterior_probs[:, 0], torch.tensor(1.0), atol=0.1)
    assert 0.0 <= posterior_probs[:, 1].item() <= 1.0 # P(Var1)
    assert 0.0 <= posterior_probs[:, 2].item() <= 1.0 # P(Var2)

    # Test conditional independence: P(Var1 | Var0) should be independent of P(Var2 | Var0)
    # if we fix Var0. This is implicitly tested by the sampling process.
    # If Var0 is True, both Var1 and Var2 should have their probabilities updated based on Var0.
    # If Var0 is False, same.

    # Check if Var1 and Var2 probabilities are different from each other (they should be, different CPTs)
    # unless embeddings are identical by chance.
    if network.num_variables == 3 : # only if Var1 and Var2 are distinct
        # This is not a strict requirement but likely with random embeddings
        if torch.isclose(posterior_probs[:, 1], posterior_probs[:, 2], atol=0.05) and \
           not torch.allclose(network.cpt_embeddings[1], network.cpt_embeddings[2]):
            print(f"Warning: P(Var1) and P(Var2) are very close with a common parent and different CPTs. "
                  f"P(Var1)={posterior_probs[:, 1].item()}, P(Var2)={posterior_probs[:, 2].item()}")


def test_topological_sort_complex_case():
    config = NeuroLiteConfig()
    config.hidden_size = 32
    config.num_bayesian_variables = 6
    # Structure: 0->1, 0->2, 1->3, 2->3, 3->4, 2->5, 4->5
    config.bayesian_network_structure = [(0,1), (0,2), (1,3), (2,3), (3,4), (2,5), (4,5)]
    network = BayesianBeliefNetwork(config=config, hidden_size=config.hidden_size)
    
    # Possible topological orders:
    # [0, 1, 2, 3, 4, 5]
    # [0, 2, 1, 3, 4, 5]
    # The calculated order must be one of these valid orders.
    order = network.topological_order
    assert len(order) == 6
    
    # Check validity of order: for every edge u->v, u must appear before v
    for u, v_list_indices in enumerate(network.children_matrix):
        if v_list_indices.sum() > 0: # If u has children
            u_pos_in_order = order.index(u)
            for v_idx_t in torch.nonzero(v_list_indices):
                v = v_idx_t.item()
                v_pos_in_order = order.index(v)
                assert u_pos_in_order < v_pos_in_order, \
                    f"Topological order violated: {u} (pos {u_pos_in_order}) not before child {v} (pos {v_pos_in_order}). Order: {order}"

def test_no_parents_case_in_get_conditional_prob(bayesian_config_predefined_chain):
    # Test _get_conditional_prob for a root node (no parents)
    # Structure: 0 -> 1 -> 2. Node 0 is a root.
    network = BayesianBeliefNetwork(config=bayesian_config_predefined_chain, hidden_size=bayesian_config_predefined_chain.hidden_size)
    batch_size = 3
    # For node 0, parent_states should be empty.
    prob_true_node0 = network._get_conditional_prob(var_idx=0, parent_states=torch.empty(batch_size, 0, device=network.cpt_embeddings.device), batch_size=batch_size)
    assert prob_true_node0.shape == (batch_size,)
    assert torch.all(prob_true_node0 >= 0) and torch.all(prob_true_node0 <= 1)
    # The actual value depends on the heuristic for priors, but it should be consistent.

# Ensure that the NeuroLiteModel correctly passes the config to BayesianBeliefNetwork
# This might require a separate integration test or checking NeuroLiteModel's __init__
# For now, this is outside the scope of testing BayesianBeliefNetwork directly.
# We assume the config object is correctly passed when used within NeuroLiteModel.
