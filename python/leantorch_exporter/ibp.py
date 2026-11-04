"""Exact (rational) Interval Bound Propagation.

Mirrors `LeanTorch.Cert.evalChainIBP`, whose soundness is a Lean theorem
(`evalChainIBP_sound`). Arithmetic is done with `fractions.Fraction`, so the bounds are exact
bounds for the *real-number* network defined by the exported weights.

Caveat: this certifies the real-valued network, not the float32/float64 inference of PyTorch
(which adds rounding error)."""
from __future__ import annotations

from fractions import Fraction
from typing import List, Sequence, Tuple

from .schema import ActivationType, LayerType, ModelSpec


class UnsupportedActivationError(ValueError):
    pass


def _matrix(layer) -> Tuple[List[List[Fraction]], List[Fraction]]:
    o, i = layer.weights.shape
    vals = [v.to_fraction() for v in layer.weights.values]
    W = [vals[r * i:(r + 1) * i] for r in range(o)]
    b = [v.to_fraction() for v in layer.bias.values] if layer.bias else [Fraction(0)] * o
    return W, b


def _linear(W, b, lo, hi):
    out_lo, out_hi = [], []
    for row, bias in zip(W, b):
        l = h = bias
        for w, a, c in zip(row, lo, hi):
            if w >= 0:
                l += w * a; h += w * c
            else:
                l += w * c; h += w * a
        out_lo.append(l); out_hi.append(h)
    return out_lo, out_hi


def _act(kind: ActivationType, slope, lo, hi):
    if kind == ActivationType.IDENTITY:
        return lo, hi
    if kind == ActivationType.RELU:
        return [max(Fraction(0), x) for x in lo], [max(Fraction(0), x) for x in hi]
    if kind == ActivationType.LEAKY_RELU:
        s = slope.to_fraction() if slope is not None else Fraction(1, 100)
        if s < 0:
            raise UnsupportedActivationError("LeakyReLU with negative slope is not monotone")
        f = lambda x: x if x >= 0 else s * x
        return [f(x) for x in lo], [f(x) for x in hi]
    raise UnsupportedActivationError(
        f"{kind.value} is not supported by the certified IBP (supported: ReLU, LeakyReLU, Identity)")


def ibp_bounds(spec: ModelSpec, lo: Sequence[Fraction], hi: Sequence[Fraction]):
    """Propagate the box [lo, hi] through the model; returns (out_lo, out_hi)."""
    lo, hi = [Fraction(x) for x in lo], [Fraction(x) for x in hi]
    for layer in spec.layers:
        if layer.layer_type == LayerType.LINEAR:
            W, b = _matrix(layer)
            if len(W[0]) != len(lo):
                raise ValueError(f"layer {layer.name}: expected {len(W[0])} inputs, got {len(lo)}")
            lo, hi = _linear(W, b, lo, hi)
        elif layer.layer_type != LayerType.ACTIVATION:
            raise UnsupportedActivationError(f"unsupported layer type {layer.layer_type.value}")
        lo, hi = _act(layer.activation, layer.negative_slope, lo, hi)
    return lo, hi


def certify_robustness(spec: ModelSpec, x0: Sequence[Fraction], eps: Fraction,
                       y_min: Sequence[Fraction], y_max: Sequence[Fraction]):
    """True iff IBP on the L∞ ball around `x0` lands inside [y_min, y_max]."""
    x0 = [Fraction(x) for x in x0]
    lo, hi = ibp_bounds(spec, [x - Fraction(eps) for x in x0], [x + Fraction(eps) for x in x0])
    ok = all(Fraction(a) <= l and h <= Fraction(b) for l, h, a, b in zip(lo, hi, y_min, y_max))
    return ok, lo, hi
