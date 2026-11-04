import LeanTorch.Core.Basic
namespace LeanTorch.Core

/-- Shape-indexed tensor: a function from the flat index range `Fin (numel shape)`. -/
def Tensor (shape : List Nat) (α : Type) : Type := Fin (numel shape) → α

namespace Tensor
variable {α β γ : Type} {shape : List Nat}

def const (shape : List Nat) (v : α) : Tensor shape α := fun _ => v
def map (f : α → β) (t : Tensor shape α) : Tensor shape β := fun i => f (t i)
def zipWith (f : α → β → γ) (t₁ : Tensor shape α) (t₂ : Tensor shape β) : Tensor shape γ :=
  fun i => f (t₁ i) (t₂ i)
def add [Add α] (t₁ t₂ : Tensor shape α) : Tensor shape α := zipWith (· + ·) t₁ t₂
def sub [Sub α] (t₁ t₂ : Tensor shape α) : Tensor shape α := zipWith (· - ·) t₁ t₂
def scale [Mul α] (c : α) (t : Tensor shape α) : Tensor shape α := map (c * ·) t

/-- Rank-1 tensors are plain vectors (definitionally). -/
abbrev Vector (n : Nat) (α : Type) : Type := Fin n → α
/-- Rank-2 tensors, stored row-major. -/
abbrev Matrix2D (m n : Nat) (α : Type) : Type := Tensor [m, n] α

def ofVector {n : Nat} (v : Vector n α) : Tensor [n] α := v
def toVector {n : Nat} (t : Tensor [n] α) : Vector n α := t

/-- Row-major index bound: `i * n + j < m * n`. -/
theorem idx_lt {m n : Nat} (i : Fin m) (j : Fin n) : i.val * n + j.val < m * n := by
  have h1 : i.val * n + j.val < i.val * n + n := Nat.add_lt_add_left j.isLt _
  have h2 : (i.val + 1) * n ≤ m * n := Nat.mul_le_mul_right n i.isLt
  have h3 : (i.val + 1) * n = i.val * n + n := Nat.succ_mul i.val n
  omega

/-- Read entry `(i, j)` of a matrix tensor. -/
def get2 {m n : Nat} (t : Matrix2D m n α) (i : Fin m) (j : Fin n) : α :=
  t ⟨i.val * n + j.val, idx_lt i j⟩

/-- Build a matrix tensor from a two-argument function. -/
def ofFun2 {m n : Nat} (f : Fin m → Fin n → α) : Matrix2D m n α := fun idx =>
  have hlt : idx.val < m * n := idx.isLt
  have hn : 0 < n := by
    rcases Nat.eq_zero_or_pos n with h | h
    · subst h; simp at hlt
    · exact h
  f ⟨idx.val / n, Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm] at hlt; exact hlt)⟩
    ⟨idx.val % n, Nat.mod_lt _ hn⟩

/-- Matrix product with statically checked inner dimension: `[m,n] × [n,p] → [m,p]`. -/
def matmul {m n p : Nat} [Add α] [Mul α] [OfNat α 0]
    (A : Matrix2D m n α) (B : Matrix2D n p α) : Matrix2D m p α :=
  ofFun2 fun i k => sumFin n fun j => get2 A i j * get2 B j k

/-- Dot product of two vectors of equal (static) length. -/
def dot {n : Nat} [Add α] [Mul α] [OfNat α 0] (v w : Vector n α) : α :=
  sumFin n fun i => v i * w i

end Tensor
end LeanTorch.Core
