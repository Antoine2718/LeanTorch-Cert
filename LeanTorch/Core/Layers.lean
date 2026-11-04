import LeanTorch.Core.Tensor
namespace LeanTorch.Core

/-- Dense layer `y = W x + b` with `W : out × in` stored as a function `Fin out → Fin in → α`. -/
structure DenseLayer (α : Type) (inD outD : Nat) where
  weights : Fin outD → Fin inD → α
  bias    : Fin outD → α

namespace DenseLayer
variable {α : Type} {inD outD : Nat}

def zero [OfNat α 0] (inD outD : Nat) : DenseLayer α inD outD := ⟨fun _ _ => 0, fun _ => 0⟩

/-- Forward pass `y_i = (∑_j W_ij * x_j) + b_i`. -/
def forward [Add α] [Mul α] [OfNat α 0] (l : DenseLayer α inD outD)
    (x : Tensor.Vector inD α) : Tensor.Vector outD α :=
  fun i => sumFin inD (fun j => l.weights i j * x j) + l.bias i

end DenseLayer

/-- Parameter-free activation layers, indexed by their width. -/
structure ReLULayer (dim : Nat) where
structure TanhLayer (dim : Nat) where
structure LeakyReLULayer (α : Type) (dim : Nat) where
  slope : α

end LeanTorch.Core
