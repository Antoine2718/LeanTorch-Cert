import Mathlib.Data.Real.Basic
import LeanTorch.Core.Tensor

namespace LeanTorch.Core

/--
Formal definition of deep neural network layer specifications with 
dependent type signatures guaranteeing dimension preservation.
-/

/-- Dense (Fully-Connected / Linear) Layer containing Weight matrix and Bias vector. -/
structure DenseLayer (in_dim out_dim : ℕ) where
  weights : Tensor.Matrix2D out_dim in_dim ℝ
  bias    : Tensor.Vector out_dim ℝ

namespace DenseLayer

/-- Forward evaluation of a Dense Layer: y = W · x + b -/
noncomputable def forward {in_dim out_dim : ℕ} 
    (layer : DenseLayer in_dim out_dim) 
    (input : Tensor.Vector in_dim ℝ) : Tensor.Vector out_dim ℝ :=
  Tensor.add (Tensor.vecMatMul input layer.weights) layer.bias

end DenseLayer

/-- Parameterless activation layers parametrized by dimension. -/
structure ReLULayer (dim : ℕ) where

namespace ReLULayer

/-- Forward evaluation of a ReLU activation layer. -/
noncomputable def forward {dim : ℕ} 
    (_ : ReLULayer dim) 
    (input : Tensor.Vector dim ℝ) : Tensor.Vector dim ℝ :=
  Tensor.relu input

end ReLULayer

structure LeakyReLULayer (dim : ℕ) where
  negative_slope : ℝ

namespace LeakyReLULayer

/-- Forward evaluation of a LeakyReLU layer. -/
noncomputable def forward {dim : ℕ} 
    (layer : LeakyReLULayer dim) 
    (input : Tensor.Vector dim ℝ) : Tensor.Vector dim ℝ :=
  Tensor.leakyRelu layer.negative_slope input

end LeakyReLULayer

structure TanhLayer (dim : ℕ) where

namespace TanhLayer

/-- Forward evaluation of a Tanh activation layer. -/
noncomputable def forward {dim : ℕ} 
    (_ : TanhLayer dim) 
    (input : Tensor.Vector dim ℝ) : Tensor.Vector dim ℝ :=
  Tensor.tanh input

end TanhLayer

end LeanTorch.Core
