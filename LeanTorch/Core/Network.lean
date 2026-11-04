import LeanTorch.Core.Layers
namespace LeanTorch.Core

/-- A well-shaped sequential network from width `i` to width `o`.
Every constructor ties the layer width to the width of the rest of the chain, so a
dimension mismatch between adjacent layers is a *type error*. -/
inductive LayerChain (α : Type) : Nat → Nat → Type where
  | nil (d : Nat) : LayerChain α d d
  | consLinear    {i m o : Nat} (layer : DenseLayer α i m)      (rest : LayerChain α m o) : LayerChain α i o
  | consReLU      {d o : Nat}   (layer : ReLULayer d)           (rest : LayerChain α d o) : LayerChain α d o
  | consLeakyReLU {d o : Nat}   (layer : LeakyReLULayer α d)    (rest : LayerChain α d o) : LayerChain α d o
  | consTanh      {d o : Nat}   (layer : TanhLayer d)           (rest : LayerChain α d o) : LayerChain α d o

namespace LayerChain
variable {α : Type} [Add α] [Mul α] [OfNat α 0] [Activations α]

/-- Forward pass. -/
def run : {i o : Nat} → LayerChain α i o → Tensor.Vector i α → Tensor.Vector o α
  | _, _, .nil _,              x => x
  | _, _, .consLinear l rest,  x => run rest (l.forward x)
  | _, _, .consReLU _ rest,    x => run rest (fun k => Activations.relu (x k))
  | _, _, .consLeakyReLU l rest, x => run rest (fun k => Activations.leakyRelu l.slope (x k))
  | _, _, .consTanh _ rest,    x => run rest (fun k => Activations.tanh (x k))

end LayerChain

namespace LayerChain
variable {α : Type}

/-- Compose two chains whose interface widths agree. -/
def concat : {i m o : Nat} → LayerChain α i m → LayerChain α m o → LayerChain α i o
  | _, _, _, .nil _,              c₂ => c₂
  | _, _, _, .consLinear l rest,  c₂ => .consLinear l (concat rest c₂)
  | _, _, _, .consReLU l rest,    c₂ => .consReLU l (concat rest c₂)
  | _, _, _, .consLeakyReLU l rest, c₂ => .consLeakyReLU l (concat rest c₂)
  | _, _, _, .consTanh l rest,    c₂ => .consTanh l (concat rest c₂)

/-- Number of layers. -/
def length : {i o : Nat} → LayerChain α i o → Nat
  | _, _, .nil _ => 0
  | _, _, .consLinear _ r => r.length + 1
  | _, _, .consReLU _ r => r.length + 1
  | _, _, .consLeakyReLU _ r => r.length + 1
  | _, _, .consTanh _ r => r.length + 1

variable [Add α] [Mul α] [OfNat α 0] [Activations α]

/-- Running a concatenation = composing the runs (proved for every scalar type). -/
theorem run_concat {i m o : Nat} (c₁ : LayerChain α i m) (c₂ : LayerChain α m o)
    (x : Tensor.Vector i α) : run (concat c₁ c₂) x = run c₂ (run c₁ x) := by
  induction c₁ with
  | nil _ => rfl
  | consLinear l rest ih => exact ih c₂ (l.forward x)
  | consReLU l rest ih => exact ih c₂ (fun k => Activations.relu (x k))
  | consLeakyReLU l rest ih => exact ih c₂ (fun k => Activations.leakyRelu l.slope (x k))
  | consTanh l rest ih => exact ih c₂ (fun k => Activations.tanh (x k))

end LayerChain

/-- Top-level wrapper. -/
structure SequentialNetwork (α : Type) (i o : Nat) where
  chain : LayerChain α i o

namespace SequentialNetwork
variable {α : Type} {i o : Nat} [Add α] [Mul α] [OfNat α 0] [Activations α]
def forward (net : SequentialNetwork α i o) (x : Tensor.Vector i α) : Tensor.Vector o α :=
  net.chain.run x
end SequentialNetwork

end LeanTorch.Core
