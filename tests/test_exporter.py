import json
import os
import tempfile
import pytest
import torch
import torch.nn as nn

from leantorch_exporter import (
    ExactRational,
    ModelExporter,
    DomainSpec,
)


def test_exact_rational_conversion():
    """Verifies that floats are accurately mapped to exact rationals."""
    r1 = ExactRational.from_float(0.5)
    assert r1.numerator == 1
    assert r1.denominator == 2
    assert r1.to_lean_string() == "((1 : ℝ) / 2)"

    r2 = ExactRational.from_float(-3.0)
    assert r2.numerator == -3
    assert r2.denominator == 1
    assert r2.to_lean_string() == "(-3 : ℝ)"


def test_model_exporter_tracing():
    """Verifies FX graph tracing and Lean code emission for a sequential model."""
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 4)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(4, 1)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    model = SimpleMLP()
    sample_input = torch.randn(1, 2)
    exporter = ModelExporter(model, sample_input)

    domain = DomainSpec(
        dimension=2,
        lower_bounds=[ExactRational.from_float(-1.0), ExactRational.from_float(-1.0)],
        upper_bounds=[ExactRational.from_float(1.0), ExactRational.from_float(1.0)],
    )

    spec = exporter.build_spec("SimpleMLP", domain=domain)
    assert spec.model_name == "SimpleMLP"
    assert spec.input_shape == [1, 2]
    assert spec.output_shape == [1, 1]
    assert len(spec.layers) == 2  # Relu is fused into activation attribute

    lean_code = exporter.generate_lean_code(spec)
    assert "def SimpleMLP_input_dim : ℕ := 2" in lean_code
    assert "def SimpleMLP_output_dim : ℕ := 1" in lean_code


def test_json_export_and_import(tmp_path):
    model = nn.Sequential(nn.Linear(3, 2))
    sample_input = torch.randn(1, 3)
    exporter = ModelExporter(model, sample_input)
    spec = exporter.build_spec("TestModel")

    json_file = tmp_path / "test_spec.json"
    exporter.export_json(spec, str(json_file))

    assert json_file.exists()
    with open(json_file, "r") as f:
        data = json.load(f)
    assert data["model_name"] == "TestModel"
