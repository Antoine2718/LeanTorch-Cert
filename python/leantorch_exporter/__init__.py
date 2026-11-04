"""LeanTorch exporter.

`schema` and `ibp` are pure Python (pydantic only). `ModelExporter` and the Lyapunov
trainer need PyTorch and are imported lazily, so `import leantorch_exporter` works
without torch installed.
"""
from .schema import (  # noqa: F401
    ActivationType, DomainSpec, ExactRational, LayerSpec, LayerType, ModelSpec, TensorSpec,
)
from .ibp import certify_robustness, ibp_bounds, UnsupportedActivationError  # noqa: F401

_LAZY = {
    "ModelExporter": "exporter",
    "UnsupportedModelError": "exporter",
    "spec_from_descriptors": "exporter",
    "LayerDescriptor": "exporter",
    "NeuralLyapunovCandidate": "lyapunov_trainer",
    "CEGISLyapunovTrainer": "lyapunov_trainer",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module 'leantorch_exporter' has no attribute {name!r}")


__version__ = "0.2.0"
