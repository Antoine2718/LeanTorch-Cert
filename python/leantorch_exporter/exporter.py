"""PyTorch -) LeanTorch exporter (straight-line sequential models only)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .schema import (ActivationType, DomainSpec, ExactRational, LayerSpec, LayerType,
                     ModelSpec, TensorSpec)

class UnsupportedModelError(ValueError):
    """The model uses an operation that LeanTorch cannot represent."""

@dataclass
class LayerDescriptor:
    """Torch-free description of one traced op (so the conversion logic is testable without torch)."""
    kind: str # "linear" | "activation"
    name: str
    weight: Optional[Sequence[Sequence[float]]] = None   # [out][in]
    bias: Optional[Sequence[float]] = None
    activation: Optional[ActivationType] = None
    negative_slope: Optional[float] = None


def _tensor(values, shape) -> TensorSpec:
    return TensorSpec(shape=list(shape), values=[ExactRational.from_float(float(v)) for v in values])


def spec_from_descriptors(descs: Sequence[LayerDescriptor], input_shape: Sequence[int],
                          model_name: str, domain: Optional[DomainSpec] = None,
                          metadata: Optional[dict] = None) -> ModelSpec:
    layers: List[LayerSpec] = []
    shape = list(input_shape)
    for d in descs:
        if d.kind == "linear":
            out_f, in_f = len(d.weight), len(d.weight[0])
            if shape[-1] != in_f:
                raise UnsupportedModelError(f"{d.name}: expects {in_f} features but input has {shape[-1]}")
            new_shape = shape[:-1] + [out_f]
            layers.append(LayerSpec(
                name=d.name, layer_type=LayerType.LINEAR, in_shape=list(shape), out_shape=new_shape,
                weights=_tensor([v for row in d.weight for v in row], (out_f, in_f)),
                bias=_tensor(d.bias, (out_f,)) if d.bias is not None else None))
            shape = new_shape
        elif d.kind == "activation":
            slope = ExactRational.from_float(d.negative_slope) if d.negative_slope is not None else None
            prev = layers[-1] if layers else None
            if prev is not None and prev.layer_type == LayerType.LINEAR \
                    and prev.activation == ActivationType.IDENTITY:
                prev.activation, prev.negative_slope = d.activation, slope   # fuse
            else:
                layers.append(LayerSpec(name=d.name, layer_type=LayerType.ACTIVATION,
                                        in_shape=list(shape), out_shape=list(shape),
                                        activation=d.activation, negative_slope=slope))
        else:
            raise UnsupportedModelError(f"unknown descriptor kind {d.kind!r}")
    return ModelSpec(model_name=model_name, input_shape=list(input_shape), output_shape=shape,
                     layers=layers, domain=domain, metadata=dict(metadata or {}))


class ModelExporter:
    """Trace a sequential `torch.nn.Module` with `torch.fx` and build a `ModelSpec`."""

    _ACT_MODULES = {"ReLU": ActivationType.RELU, "LeakyReLU": ActivationType.LEAKY_RELU,
                    "Tanh": ActivationType.TANH, "Sigmoid": ActivationType.SIGMOID}

    def __init__(self, model, sample_input):
        import torch  # noqa: F401  (fail early with a clear message if torch is missing)
        self.model = model.eval()
        self.sample_input = sample_input

    def _descriptors(self) -> List[LayerDescriptor]:
        import torch
        import torch.nn.functional as F
        gm = torch.fx.symbolic_trace(self.model)
        descs: List[LayerDescriptor] = []
        prev = None
        for node in gm.graph.nodes:
            if node.op in ("placeholder", "output"):
                prev = node if node.op == "placeholder" else prev
                continue
            if not node.args or node.args[0] is not prev:
                raise UnsupportedModelError(f"node {node.name}: only straight-line sequential models are supported")
            if node.op == "call_module":
                mod = gm.get_submodule(node.target)
                cls = type(mod).__name__
                if isinstance(mod, torch.nn.Linear):
                    descs.append(LayerDescriptor(
                        "linear", node.target, mod.weight.detach().double().tolist(),
                        mod.bias.detach().double().tolist() if mod.bias is not None else None))
                elif cls in self._ACT_MODULES:
                    slope = float(mod.negative_slope) if cls == "LeakyReLU" else None
                    descs.append(LayerDescriptor("activation", node.target,
                                                 activation=self._ACT_MODULES[cls], negative_slope=slope))
                else:
                    raise UnsupportedModelError(f"module {node.target} ({cls}) is not supported")
            elif node.op == "call_function" and node.target in (torch.relu, F.relu):
                descs.append(LayerDescriptor("activation", node.name, activation=ActivationType.RELU))
            elif node.op == "call_function" and node.target in (torch.tanh, F.tanh):
                descs.append(LayerDescriptor("activation", node.name, activation=ActivationType.TANH))
            else:
                raise UnsupportedModelError(f"operation {node.op}:{node.target} is not supported")
            prev = node
        return descs

    def build_spec(self, model_name: str, domain: Optional[DomainSpec] = None) -> ModelSpec:
        return spec_from_descriptors(self._descriptors(), list(self.sample_input.shape), model_name,
                                     domain, {"exporter": "leantorch_exporter 0.2.0"})

    def generate_lean_code(self, spec: ModelSpec, json_path: Optional[str] = None) -> str:
        return generate_lean_code(spec, json_path)


def generate_lean_code(spec: ModelSpec, json_path: Optional[str] = None) -> str:
    """Lean file that imports the model from its JSON (single source of truth)."""
    path = json_path or f"{spec.model_name}.json"
    n = spec.model_name
    return f"""import LeanTorch
/-! Auto-generated by leantorch_exporter. Run `ModelSpec.write_json("{path}")` first. -/
open LeanTorch.Core

def {n}_input_dim : ℕ := {spec.input_shape[-1]}
def {n}_output_dim : ℕ := {spec.output_shape[-1]}
