/-!
LeanTorch.Core.Basic

Mathlib-free foundations: shape arithmetic, flat/multi index conversion, a generic
finite sum and the scalar-activation interface.  Everything here is computable, so the
same definitions can be *executed* (over `Float`) and *proved about* (over `ℝ`, in `Cert`).
-/
namespace LeanTorch.Core

/-- Number of elements of a tensor of shape `shape`.
Definitionally `numel [n] = n`, so a rank-1 tensor *is* a `Fin n → α`. -/
def numel : List Nat → Nat
  | []        => 1
  | [d]       => d
  | d :: ds   => d * numel ds

/-- Row-major strides: `strides [2,3,4] = [12,4,1]`. -/
def strides : List Nat → List Nat
  | []      => []
  | _ :: ds => numel ds :: strides ds

/-- Multi-index → flat (row-major) index. -/
def multiToFlat (shape coords : List Nat) : Nat :=
  ((strides shape).zip coords).foldl (fun acc sc => acc + sc.1 * sc.2) 0

/-- Flat (row-major) index → multi-index. -/
def flatToMulti : List Nat → Nat → List Nat
  | [], _ => []
  | _ :: ds, idx =>
    let s := numel ds
    if s = 0 then 0 :: flatToMulti ds 0
    else (idx / s) :: flatToMulti ds (idx % s)

example : strides [2, 3, 4] = [12, 4, 1] := by decide
example : multiToFlat [2, 3, 4] [1, 2, 3] = 23 := by decide
example : flatToMulti [2, 3, 4] 23 = [1, 2, 3] := by decide

/-- Generic finite sum `∑ i : Fin n, f i` (needs only `+` and `0`). -/
def sumFin {α : Type} [Add α] [OfNat α 0] : (n : Nat) → (Fin n → α) → α
  | 0,     _ => 0
  | n + 1, f => sumFin n (fun i => f i.castSucc) + f (Fin.last n)

/-- Scalar activation functions.  Instances: `Float` (executable, in `Core.Float`)
and `ℝ` (in `Cert.RealInstances`, used for proofs). -/
class Activations (α : Type) where
  relu      : α → α
  leakyRelu : α → α → α   -- slope, value
  tanh      : α → α

end LeanTorch.Core
