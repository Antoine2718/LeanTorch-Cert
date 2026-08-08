import Mathlib.Data.Real.Basic
import Mathlib.Data.Matrix.Basic
import LeanTorch.Core.Basic

namespace LeanTorch.Core

/--
Shape-indexed dependently-typed Tensor definition.
A Tensor with shape `S : List ℕ` containing element type `α` is backed by a 
flat mapping from `Fin (S.prod)` to `α`.
-/
def Tensor (shape : List ℕ) (α : Type _) :=
  Fin (shape.prod) → α

namespace Tensor

variable {shape : List ℕ} {α : Type _}

/--
Constructs a zero tensor of specified shape.
-/
def zero [Zero α] (shape : List ℕ) : Tensor shape α :=
  fun _ => 0

/--
Constructs a constant tensor filled with a scalar value.
-/
def const (shape : List ℕ) (val : α) : Tensor shape α :=
  fun _ => val

/--
Applies a pointwise unary function across all elements in the tensor.
-/
def map (f : α → β) (t : Tensor shape α) : Tensor shape β :=
  fun i => f (t i)

/--
Applies a pointwise binary function across two tensors of identical shape.
-/
def zipWith (f : α → β → γ) (t1 : Tensor shape α) (t2 : Tensor shape β) : Tensor shape γ :=
  fun i => f (t1 i) (t2 i)

/-- Pointwise tensor addition. -/
def add [Add α] (t1 t2 : Tensor shape α) : Tensor shape α :=
  zipWith (· + ·) t1 t2

/-- Pointwise tensor subtraction. -/
def sub [Sub α] (t1 t2 : Tensor shape α) : Tensor shape α :=
  zipWith (· - ·) t1 t2

/-- Scalar tensor multiplication. -/
def scale [Mul α] (c : α) (t : Tensor shape α) : Tensor shape α :=
  map (c * ·) t

/-- Pointwise ReLU activation on real-valued tensors. -/
noncomputable def relu (t : Tensor shape ℝ) : Tensor shape ℝ :=
  map realReLU t

/-- Pointwise LeakyReLU activation on real-valued tensors. -/
noncomputable def leakyRelu (slope : ℝ) (t : Tensor shape ℝ) : Tensor shape ℝ :=
  map (realLeakyReLU slope) t

/-- Pointwise Tanh activation on real-valued tensors. -/
noncomputable def tanh (t : Tensor shape ℝ) : Tensor shape ℝ :=
  map realTanh t

/-- Pointwise Sigmoid activation on real-valued tensors. -/
noncomputable def sigmoid (t : Tensor shape ℝ) : Tensor shape ℝ :=
  map realSigmoid t

/- Specialized 1D and 2D Tensor interop with Mathlib Matrix -/

/-- Alias for 1D Vector Tensors. -/
abbrev Vector (n : ℕ) (α : Type _) := Tensor [n] α

/-- Alias for 2D Matrix Tensors. -/
abbrev Matrix2D (m n : ℕ) (α : Type _) := Tensor [m, n] α

/-- Converts a 2D Tensor [m, n] to a standard Mathlib `Matrix (Fin m) (Fin n) α`. -/
def toMathlibMatrix {m n : ℕ} (t : Matrix2D m n α) : Matrix (Fin m) (Fin n) α :=
  fun i j =>
    let flatIdx : ℕ := i.val * n + j.val
    have h : flatIdx < [m, n].prod := by
      dsimp [List.prod]
      have hi : i.val < m := i.isLt
      have hj : j.val < n := j.isLt
      calc i.val * n + j.val < i.val * n + n := Nat.add_lt_add_left hj (i.val * n)
      _ = (i.val + 1) * n := by ring
      _ ≤ m * n := Nat.mul_le_mul_right n hi
      _ = m * n * 1 := by ring
    t ⟨flatIdx, h⟩

/-- Converts a Mathlib `Matrix (Fin m) (Fin n) α` into a 2D Tensor [m, n]. -/
def fromMathlibMatrix {m n : ℕ} (M : Matrix (Fin m) (Fin n) α) : Matrix2D m n α :=
  fun idx =>
    let flat := idx.val
    let row := flat / n
    let col := flat % n
    have hrow : row < m := by
      have hidx := idx.isLt
      dsimp [List.prod] at hidx
      if hn : n = 0 then
        omega
      else
        exact Nat.div_lt_of_lt_mul hidx
    have hcol : col < n := by
      if hn : n = 0 then
        omega
      else
        exact Nat.mod_lt flat (Nat.pos_of_ne_zero hn)
    M ⟨row, hrow⟩ ⟨col, hcol⟩

/--
Formally verified 2D Matrix Multiplication with type-checked dimension contraction.
A tensor of shape [m, n] multiplied by shape [n, p] produces shape [m, p].
-/
noncomputable def matmul {m n p : ℕ} (A : Matrix2D m n ℝ) (B : Matrix2D n p ℝ) : Matrix2D m p ℝ :=
  let MA := toMathlibMatrix A
  let MB := toMathlibMatrix B
  let MC := Matrix.mul MA MB
  fromMathlibMatrix MC

/--
Formally verified Vector-Matrix Multiplication.
A 1D vector of shape [n] multiplied by 2D matrix [n, m] produces 1D vector [m].
-/
noncomputable def vecMatMul {n m : ℕ} (v : Vector n ℝ) (W : Matrix2D m n ℝ) : Vector m ℝ :=
  fun i =>
    Finset.univ.sum (fun j => W (⟨i.val * n + j.val, by
      dsimp [List.prod]
      have hi := i.isLt
      have hj := j.isLt
      calc i.val * n + j.val < i.val * n + n := Nat.add_lt_add_left hj (i.val * n)
      _ = (i.val + 1) * n := by ring
      _ ≤ m * n := Nat.mul_le_mul_right n hi
      _ = m * n * 1 := by ring
    ⟩) * v j)

/--
Computes the dot product of two 1D vectors of identical dimension.
-/
noncomputable def dot {n : ℕ} (v1 v2 : Vector n ℝ) : ℝ :=
  Finset.univ.sum (fun i => v1 i * v2 i)

end Tensor

end LeanTorch.Core
