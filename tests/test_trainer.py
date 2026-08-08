import pytest
import torch
from leantorch_exporter import CEGISLyapunovTrainer, NeuralLyapunovCandidate


def test_neural_lyapunov_candidate_origin_zero():
    """Verifies that candidate V_θ(x) strictly satisfies V_θ(0) = 0."""
    model = NeuralLyapunovCandidate(state_dim=2, hidden_dim=16)
    origin = torch.zeros(1, 2)
    v_origin = model(origin)
    assert torch.abs(v_origin).item() < 1e-6


def test_cegis_trainer_iteration():
    """Tests a single training iteration and falsifier counterexample search."""
    # 2D Linear stable system: x1_dot = -x1, x2_dot = -x2
    def stable_dynamics(x):
        return -x

    trainer = CEGISLyapunovTrainer(
        dynamics=stable_dynamics,
        state_dim=2,
        domain_radius=1.5,
        device="cpu",
    )

    metrics = trainer.train_step()
    assert "total_loss" in metrics
    assert "max_lie_violation" in metrics

    counterexamples = trainer.find_counterexamples(num_searches=50, steps=10)
    assert isinstance(counterexamples, torch.Tensor)
