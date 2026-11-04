import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn
from leantorch_exporter import DomainSpec, ExactRational, ModelExporter
from leantorch_exporter.exporter import UnsupportedModelError


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.relu, self.fc2 = nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_model_exporter_tracing():
    model = SimpleMLP()
    ex = ModelExporter(model, torch.randn(1, 2))
    domain = DomainSpec(dimension=2, lower_bounds=[ExactRational.from_float(-1.0)] * 2,
                        upper_bounds=[ExactRational.from_float(1.0)] * 2)
    spec = ex.build_spec("SimpleMLP", domain=domain)
    assert spec.input_shape == [1, 2] and spec.output_shape == [1, 1] and len(spec.layers) == 2
    code = ex.generate_lean_code(spec)
    assert "def SimpleMLP_input_dim : ℕ := 2" in code and "def SimpleMLP_output_dim : ℕ := 1" in code


def test_exported_weights_are_exact():
    model = SimpleMLP()
    spec = ModelExporter(model, torch.randn(1, 2)).build_spec("m")
    w = model.fc1.weight.detach().double().flatten().tolist()
    assert [float(v.to_fraction()) for v in spec.layers[0].weights.values] == w


def test_sequential_sequence_and_unsupported_ops():
    seq = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    assert len(ModelExporter(seq, torch.randn(1, 2)).build_spec("s").layers) == 2
    class Skip(nn.Module):
        def __init__(self): super().__init__(); self.f = nn.Linear(2, 2)
        def forward(self, x): return self.f(x) + x
    with pytest.raises(UnsupportedModelError):
        ModelExporter(Skip(), torch.randn(1, 2)).build_spec("skip")
