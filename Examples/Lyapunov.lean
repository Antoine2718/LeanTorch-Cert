import LeanTorch
/-!
# Example 3 — Lyapunov conditions for a damped oscillator
  ẋ₁ = x₂,  ẋ₂ = -x₁ - x₂

`V(x) = ³⁄₂x₁² + x₁x₂ + x₂²` gives `V̇ = -x₁² - x₂² < 0` for `x ≠ 0`, so *strict* decrease holds.
(The naive choice `V = x₁² + x₂²` only gives `V̇ = -2x₂² ≤ 0`, which is not strict.)
-/
open LeanTorch.Core LeanTorch.Cert

namespace Examples.Lyapunov

def osc : DynamicalSystem 2 :=
  ⟨fun x i => match i with
    | ⟨0, _⟩ => x 1
    | _      => -x 0 - x 1⟩

def V (x : Fin 2 → ℝ) : ℝ := 3 / 2 * (x 0) ^ 2 + x 0 * x 1 + (x 1) ^ 2

def gradV (x : Fin 2 → ℝ) : Fin 2 → ℝ := fun i =>
  match i with
  | ⟨0, _⟩ => 3 * x 0 + x 1
  | _      => x 0 + 2 * x 1

def box : IntervalVector 2 := fun _ => ⟨-2, 2, by norm_num⟩

theorem lie_eq (x : Fin 2 → ℝ) : lieDerivative osc gradV x = -(x 0) ^ 2 - (x 1) ^ 2 := by
  show 0 + (3 * x 0 + x 1) * x 1 + (x 0 + 2 * x 1) * (-x 0 - x 1) = _
  ring

theorem exists_coord_ne {x : Fin 2 → ℝ} (h : x ≠ fun _ => 0) : x 0 ≠ 0 ∨ x 1 ≠ 0 := by
  by_contra hc
  push_neg at hc
  apply h
  funext i
  match i with
  | ⟨0, _⟩ => exact hc.1
  | ⟨1, _⟩ => exact hc.2

theorem osc_conditions : LyapunovConditions osc V gradV box where
  origin_zero := by simp [V]
  pos_def := by
    intro x _ hne
    unfold V
    rcases exists_coord_ne hne with h | h
    · have : 0 < (x 0) ^ 2 := by positivity
      nlinarith [sq_nonneg (x 0 + x 1), sq_nonneg (x 1)]
    · have : 0 < (x 1) ^ 2 := by positivity
      nlinarith [sq_nonneg (x 0 + x 1), sq_nonneg (x 0)]
  neg_lie := by
    intro x _ hne
    rw [lie_eq]
    rcases exists_coord_ne hne with h | h
    · have : 0 < (x 0) ^ 2 := by positivity
      nlinarith [sq_nonneg (x 1)]
    · have : 0 < (x 1) ^ 2 := by positivity
      nlinarith [sq_nonneg (x 0)]

end Examples.Lyapunov
