"""
Core Intermediate Representation (IR) schemas for LeanTorch-Cert.
Defines strongly-typed data structures for neural network topologies,
exact numerical weights (rationals), and verification domain specifications.
"""
from __future__ import annotations

from enum import Enum
from fractions import Fraction
from typing import List, Tuple, Dict, Optional, Union, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator


class ActivationType(str, Enum):
    IDENTITY = "Identity"
    RELU = "ReLU"
    LEAKY_RELU = "LeakyReLU"
    TANH = "Tanh"
    SIGMOID = "Sigmoid"
    SQUARE = "Square"


class LayerType(str, Enum):
    LINEAR = "Linear"
    CONV2D = "Conv2D"
    FLATTEN = "Flatten"
    ACTIVATION = "Activation"


class ExactRational(BaseModel):
    """
    Represents a real number as an exact rational number (q = num / den)
    to eliminate IEEE-754 floating-point ambiguity during Lean 4 ingestion.
    """
    model_config = ConfigDict(frozen=True)

    numerator: int
    denominator: int = Field(gt=0)

    @classmethod
    from_float(cls, value: float, max_denominator: int = 1_000_000_000) -> ExactRational:
        """
        Converts a float to an exact rational representation.
        Uses exact dyadic/fraction decomposition with optional denominator capping.
        """
        frac = Fraction(value).limit_denominator(max_denominator)
        return cls(numerator=frac.numerator, denominator=frac.denominator)

    def to_lean_string(self) -> str:
        """Renders the rational as a Lean 4 Real term ((num : ℝ) / den)."""
        if self.denominator == 1:
            return f"({self.numerator} : ℝ)"
        return f"(({self.numerator} : ℝ) / {self.denominator})"


class TensorSpec(BaseModel):
    """
    Specifies a shape-indexed tensor with exact rational entries.
    """
    shape: List[int] = Field(..., description="Dimensions of the tensor [d_1, d_2, ..., d_k]")
    values: List[ExactRational] = Field(..., description="Flattened list of exact rational elements")

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        res = 1
        for d in self.shape:
            res *= d
        return res

    @model_validator(mode="after")
    def validate_element_count(self) -> TensorSpec:
        if len(self.values) != self.numel:
            raise ValueError(
                f"Tensor numel mismatch: shape {self.shape} requires {self.numel} "
                f"elements, but got {len(self.values)}"
            )
        return self


class LayerSpec(BaseModel):
    """
    Formal specification of a single layer within the computational graph.
    """
    name: str
    layer_type: LayerType
    in_shape: List[int]
    out_shape: List[int]
    weights: Optional[TensorSpec] = None
    bias: Optional[TensorSpec] = None
    activation: ActivationType = ActivationType.IDENTITY
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class DomainSpec(BaseModel):
    """
    Defines a compact hyper-rectangular state-space domain D ⊂ ℝⁿ.
    D = { x ∈ ℝⁿ | lower_bounds[i] <= x[i] <= upper_bounds[i] }
    """
    dimension: int
    lower_bounds: List[ExactRational]
    upper_bounds: List[ExactRational]

    @model_validator(mode="after")
    def validate_bounds(self) -> DomainSpec:
        if len(self.lower_bounds) != self.dimension or len(self.upper_bounds) != self.dimension:
            raise ValueError("Domain bound dimensions must match specified state space dimension.")
        for i in range(self.dimension):
            lb = self.lower_bounds[i].numerator / self.lower_bounds[i].denominator
            ub = self.upper_bounds[i].numerator / self.upper_bounds[i].denominator
            if lb >= ub:
                raise ValueError(f"Lower bound at dim {i} ({lb}) must be strictly less than upper bound ({ub}).")
        return self


class ModelSpec(BaseModel):
    """
    Complete exported specification of a neural network ready for Lean 4 verification.
    """
    model_name: str
    input_shape: List[int]
    output_shape: List[int]
    layers: List[LayerSpec]
    domain: Optional[DomainSpec] = None
    metadata: Dict[str, str] = Field(default_factory=dict)
