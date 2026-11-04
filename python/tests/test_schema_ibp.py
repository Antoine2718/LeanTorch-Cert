import json, random
from fractions import Fraction
import pytest
from leantorch_exporter import (ActivationType, DomainSpec, ExactRational, UnsupportedActivationError,
                                certify_robustness, ibp_bounds)
from leantorch_exporter.exporter import (LayerDescriptor, UnsupportedModelError, generate_lean_code,
                                         spec_from_descriptors)

W1 = [[0.5, -1.0], [0.0, 0.75]]; B1 = [0.25, -0.5]; W2 = [[2.0, 1.0]]

def mlp(act=ActivationType.RELU, slope=None):
    return spec_from_descriptors([
        LayerDescriptor("linear", "fc1", W1, B1),
        LayerDescriptor("activation", "a", activation=act, negative_slope=slope),
        LayerDescriptor("linear", "fc2", W2, None)], [1, 2], "mlp")

def exact_forward(spec, x):
    x = [Fraction(v) for v in x]
    for l in spec.layers:
        if l.layer_type.value == "Linear":
            o, i = l.weights.shape; v = [t.to_fraction() for t in l.weights.values]
            b = [t.to_fraction() for t in l.bias.values] if l.bias else [0] * o
            x = [sum(v[r * i + c] * x[c] for c in range(i)) + b[r] for r in range(o)]
        if l.activation == ActivationType.RELU: x = [max(Fraction(0), t) for t in x]
        if l.activation == ActivationType.LEAKY_RELU:
            s = l.negative_slope.to_fraction(); x = [t if t >= 0 else s * t for t in x]
    return x

def test_exact_rational():
    r = ExactRational.from_float(0.5)
    assert (r.numerator, r.denominator) == (1, 2) and r.to_lean_string() == "((1 : ℝ) / 2)"
    r = ExactRational.from_float(-3.0)
    assert r.to_lean_string() == "(-3 : ℝ)"
    x = 0.1                                   # not 1/10: the exact value of the double
    assert ExactRational.from_float(x).to_fraction() == Fraction(x) != Fraction(1, 10)
    with pytest.raises(ValueError): ExactRational.from_float(float("nan"))

def test_domain_validation():
    with pytest.raises(ValueError):
        DomainSpec(dimension=1, lower_bounds=[ExactRational.from_float(1.0)], upper_bounds=[ExactRational.from_float(1.0)])

def test_spec_fusion_and_shapes():
    s = mlp()
    assert len(s.layers) == 2 and s.layers[0].activation == ActivationType.RELU
    assert s.output_shape == [1, 1]
    with pytest.raises(UnsupportedModelError):
        spec_from_descriptors([LayerDescriptor("linear", "x", [[1.0, 2.0, 3.0]], None)], [1, 2], "bad")

def test_json_roundtrip_matches_lean_schema(tmp_path):
    p = tmp_path / "m.json"; mlp().write_json(str(p))
    d = json.loads(p.read_text())
    assert d["layers"][0]["activation"] == "ReLU" and d["layers"][0]["weights"]["shape"] == [2, 2]
    assert "#import_torch_model_noncomputable" in generate_lean_code(mlp(), "m.json")

@pytest.mark.parametrize("act,slope", [(ActivationType.RELU, None), (ActivationType.LEAKY_RELU, 0.1)])
def test_ibp_is_sound_on_random_points(act, slope):
    s = mlp(act, slope); rnd = random.Random(0)
    for _ in range(40):
        x0 = [Fraction(rnd.randint(-8, 8), 4) for _ in range(2)]; eps = Fraction(rnd.randint(1, 6), 8)
        lo, hi = ibp_bounds(s, [a - eps for a in x0], [a + eps for a in x0])
        for _ in range(25):
            x = [a + eps * Fraction(rnd.randint(-8, 8), 8) for a in x0]
            y = exact_forward(s, x)
            assert all(l <= v <= h for l, v, h in zip(lo, y, hi))

def test_certify_robustness_verdicts():
    s = mlp()
    ok, lo, hi = certify_robustness(s, [1, 2], Fraction(1, 100), [Fraction(0)], [Fraction(3)])
    assert ok and lo[0] <= exact_forward(s, [1, 2])[0] <= hi[0]
    ok2, *_ = certify_robustness(s, [1, 2], Fraction(1, 100), [Fraction(0)], [Fraction(1, 2)])
    assert not ok2

def test_tanh_is_rejected_like_in_lean():
    with pytest.raises(UnsupportedActivationError):
        ibp_bounds(mlp(ActivationType.TANH), [Fraction(0)] * 2, [Fraction(1)] * 2)
