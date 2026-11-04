import pytest
torch = pytest.importorskip("torch")
from leantorch_exporter import CEGISLyapunovTrainer, NeuralLyapunovCandidate


def test_candidate_origin_zero():
    m = NeuralLyapunovCandidate(state_dim=2, hidden_dim=16)
    assert torch.abs(m(torch.zeros(1, 2))).item() < 1e-6


def test_cegis_trainer_iteration():
    tr = CEGISLyapunovTrainer(dynamics=lambda x: -x, state_dim=2, domain_radius=1.5, device="cpu")
    m = tr.train_step()
    assert "total_loss" in m and "max_lie_violation" in m
    cex = tr.find_counterexamples(num_searches=50, steps=10)
    assert isinstance(cex, torch.Tensor) and cex.ndim == 2 and cex.shape[1] == 2


def test_falsifier_moves_points_and_stays_in_domain():
    tr = CEGISLyapunovTrainer(dynamics=lambda x: -x, state_dim=2, domain_radius=1.0)
    cex = tr.find_counterexamples(num_searches=64, steps=20)   # untrained V ⇒ violations exist
    assert cex.shape[0] > 0 and bool((cex.norm(dim=1) <= 1.0 + 1e-5).all())


def test_training_reduces_violation_on_stable_system():
    torch.manual_seed(0)
    tr = CEGISLyapunovTrainer(dynamics=lambda x: -x, state_dim=2, domain_radius=1.0, hidden_dim=16)
    first = tr.train_step()["max_lie_violation"]
    for _ in range(300): last = tr.train_step()["max_lie_violation"]
    assert last < first
