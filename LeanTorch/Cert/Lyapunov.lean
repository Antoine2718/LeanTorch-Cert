import LeanTorch.Cert.Semantics
namespace LeanTorch.Cert
noncomputable section
open LeanTorch.Core

/-- Autonomous system `ẋ = f(x)` on `ℝⁿ`. -/
structure DynamicalSystem (n : Nat) where
  vectorField : (Fin n → ℝ) → (Fin n → ℝ)

/-- `⟨∇V(x), f(x)⟩` for a *supplied* gradient field `gradV`. -/
noncomputable def lieDerivative {n : Nat} (sys : DynamicalSystem n)
    (gradV : (Fin n → ℝ) → (Fin n → ℝ)) (x : Fin n → ℝ) : ℝ :=
  sumFin n (fun i => gradV x i * sys.vectorField x i)

/-- The pointwise Lyapunov inequalities on a box `D`:
`V(0) = 0`, `V > 0` and `⟨∇V, f⟩ < 0` on `D \ {0}`.

**Scope note.** This structure records the *conditions* of Lyapunov's theorem. It does not
by itself prove stability, and Lean does not check that `gradV` really is `∇V`
(that link is the user's obligation in this version). -/
structure LyapunovConditions {n : Nat} (sys : DynamicalSystem n) (V : (Fin n → ℝ) → ℝ)
    (gradV : (Fin n → ℝ) → (Fin n → ℝ)) (domain : IntervalVector n) : Prop where
  origin_zero : V (fun _ => 0) = 0
  pos_def : ∀ x, domain.contains x → x ≠ (fun _ => 0) → 0 < V x
  neg_lie : ∀ x, domain.contains x → x ≠ (fun _ => 0) → lieDerivative sys gradV x < 0

end -- noncomputable section
end LeanTorch.Cert
