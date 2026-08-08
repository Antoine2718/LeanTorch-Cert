import Mathlib.Data.Real.Basic
import LeanTorch.Core.Tensor
import LeanTorch.Core.Layers

namespace LeanTorch.Core

/--
Dependently-typed inductive GADT representing a valid sequential neural network architecture.
Guarantees at compile time that layer input/output shape invariants strictly chain together.
An invalid dimension match causes a Lean compilation error.
-/
inductive LayerChain : ℕ → ℕ → Type where
  | nil (dim : ℕ) : LayerChain dim dim
  | consLinear {in_dim mid_dim out_dim : ℕ} 
      (layer : DenseLayer in_dim mid_dim) 
      (rest : LayerChain mid_dim out_dim) : LayerChain in_dim out_dim
  | consReLU {dim out_dim : ℕ} 
      (layer : ReLULayer dim) 
      (rest : LayerChain dim out_dim) : LayerChain in_dim out_dim
  | consLeakyReLU {dim out_dim : ℕ} 
      (layer : LeakyReLULayer dim) 
      (rest : LayerChain dim out_dim) : LayerChain in_dim out_dim
  | consTanh {dim out_dim : ℕ} 
      (layer : TanhLayer dim) 
      (rest : LayerChain dim out_dim) : LayerChain in_dim out_dim

namespace LayerChain

/--
Executes a forward pass through the dependently-typed layer chain.
-/
noncomputable def run {in_dim out_dim : ℕ} 
    (chain : LayerChain in_dim out_dim) 
    (input : Tensor.Vector in_dim ℝ) : Tensor.Vector out_dim ℝ :=
  match chain with
  | nil _ => input
  | consLinear layer rest =>
    let mid := DenseLayer.forward layer input
    run rest mid
  | consReLU layer rest =>
    let mid := ReLULayer.forward layer input
    run rest mid
  | consLeakyReLU layer rest =>
    let mid := LeakyReLULayer.forward layer input
    run rest mid
  | consTanh layer rest =>
    let mid := TanhLayer.forward layer input
    run rest mid

/--
Concatenates two compatible layer chains into a single combined network chain.
-/
def concat {in_dim mid_dim out_dim : ℕ}
    (c1 : LayerChain in_dim mid_dim)
    (c2 : LayerChain mid_dim out_dim) : LayerChain in_dim out_dim :=
  match c1 with
  | nil _ => c2
  | consLinear layer rest => consLinear layer (concat rest c2)
  | consReLU layer rest => consReLU layer (concat rest c2)
  | consLeakyReLU layer rest => consLeakyReLU layer (concat rest c2)
  | consTanh layer rest => consTanh layer (concat rest c2)

/--
Theorem: Forward pass evaluation over concatenated chains equals sequential function composition.
-/
theorem run_concat_eq_composition {in_dim mid_dim out_dim : ℕ}
    (c1 : LayerChain in_dim mid_dim)
    (c2 : LayerChain mid_dim out_dim)
    (x : Tensor.Vector in_dim ℝ) :
    run (concat c1 c2) x = run c2 (run c1 x) := by
  induction c1 with
  | nil _ => rfl
  | consLinear layer rest ih =>
    dsimp [concat, run]
    rw [ih]
  | consReLU layer rest ih =>
    dsimp [concat, run]
    rw [ih]
  | consLeakyReLU layer rest ih =>
    dsimp [concat, run]
    rw [ih]
  | consTanh layer rest ih =>
    dsimp [concat, run]
    rw [ih]

end LayerChain

/--
Top-level wrapper structure for a verified sequential neural network.
-/
structure SequentialNetwork (in_dim out_dim : ℕ) where
  chain : LayerChain in_dim out_dim

namespace SequentialNetwork

/-- Evaluates the sequential network on an input vector. -/
noncomputable def forward {in_dim out_dim : ℕ}
    (net : SequentialNetwork in_dim out_dim)
    (input : Tensor.Vector in_dim ℝ) : Tensor.Vector out_dim ℝ :=
  LayerChain.run net.chain input

end SequentialNetwork

end LeanTorch.Core
