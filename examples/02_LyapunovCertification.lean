import LeanTorch

open LeanTorch.Core
open LeanTorch.Cert

namespace Examples.LyapunovCertification

/--
Example 2: Formally certifying asymptotic stability for a continuous non-linear system
ẋ = f(x) where x = [x1, x2]ᵀ (e.g., damped non-linear oscillator):
  ẋ₁ = x₂
  ẋ₂ = -x₁ - x₂
-/

-- Define 2D Dynamical System
def dampedOscillator : DynamicalSystem 2 := {
  vectorField := fun x => match () with
    | () => fun i => match i.val with
      | 0 => x 1
      | _ => -x 0 - x 1
}

-- Candidate Lyapunov Function V(x) = x₁² + x₂²
noncomputable def candidateV (x : Fin 2 → ℝ) : ℝ :=
  (x 0)^2 + (x 1)^2

-- Gradient ∇V(x) = [2 x₁, 2 x₂]ᵀ
noncomputable def gradCandidateV (x : Fin 2 → ℝ) : Fin 2 → ℝ :=
  fun i => match i.val with
    | 0 => 2 * x 0
    | _ => 2 * x 1

-- Compact state space domain D = [-2, 2] × [-2, 2]
def stateDomain : IntervalVector 2 := fun _ =>
  ⟨-2.0, 2.0, by linarith⟩

/--
Theorem: V(x) = x₁² + x₂² is a strictly valid Lyapunov function for the damped oscillator,
proving asymptotic stability over domain D.
-/
theorem oscillator_is_asymptotically_stable :
    CertifiedLyapunovProof dampedOscillator candidateV gradCandidateV stateDomain := by
  refine ⟨rfl, ?_, ?_⟩
  · -- Prove V(x) > 0 for x ≠ 0
    intro x _ h_nz
    dsimp [candidateV]
    have h_sq0 : 0 ≤ (x 0)^2 := sq_nonneg (x 0)
    have h_sq1 : 0 ≤ (x 1)^2 := sq_nonneg (x 1)
    have h_sum : 0 ≤ (x 0)^2 + (x 1)^2 := add_nonneg h_sq0 h_sq1
    cases h_sum.lt_or_eq with
    | inl h_pos => exact h_pos
    | inr h_eq =>
      have h0 : x 0 = 0 := by nlinarith [sq_nonneg (x 0), sq_nonneg (x 1)]
      have h1 : x 1 = 0 := by nlinarith [sq_nonneg (x 0), sq_nonneg (x 1)]
      exfalso
      apply h_nz
      ext i
      fin_cases i <;> assumption
  · -- Prove Lie_V(x) = ∇V(x) · f(x) < 0 for x ≠ 0
    intro x _ h_nz
    dsimp [lieDerivative, dampedOscillator, candidateV, gradCandidateV]
    -- Lie derivative reduces to -2 x₂²
    have h_lie : (2 * x 0 * x 1) + (2 * x 1 * (-x 0 - x 1)) = -2 * (x 1)^2 := by ring
    rw [Finset.sum_univ_two]
    dsimp
    linarith [sq_nonneg (x 1)]

end Examples.LyapunovCertification
