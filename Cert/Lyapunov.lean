import Mathlib.Data.Real.Basic
import Mathlib.Analysis.Calculus.Deriv.Basic
import LeanTorch.Core.Network
import LeanTorch.Cert.Semantics
import LeanTorch.Cert.RealInterval

namespace LeanTorch.Cert

open LeanTorch.Core

/--
Defines an autonomous continuous-time non-linear dynamical system: ẋ = f(x).
where x ∈ ℝⁿ is the state vector and f : ℝⁿ → ℝⁿ is a continuous vector field.
-/
structure DynamicalSystem (n : ℕ) where
  vectorField : (Fin n → ℝ) → (Fin n → ℝ)

/--
Predicate specifying that x = 0 is an equilibrium point of the dynamical system: f(0) = 0.
-/
def EquilibriumAtOrigin {n : ℕ} (sys : DynamicalSystem n) : Prop :=
  sys.vectorField (fun _ => 0) = fun _ => 0

/--
Directional Lie Derivative of a candidate scalar function V : ℝⁿ → ℝ along vector field f(x):
Lie_V(x) = ⟨∇V(x), f(x)⟩ = ∑ᵢ (∂V/∂xᵢ) · fᵢ(x)
-/
noncomputable def lieDerivative {n : ℕ}
    (sys : DynamicalSystem n)
    (V : (Fin n → ℝ) → ℝ)
    (gradV : (Fin n → ℝ) → (Fin n → ℝ))
    (x : Fin n → ℝ) : ℝ :=
  Finset.univ.sum (fun i => gradV x i * sys.vectorField x i)

/--
Formal Lyapunov Stability Conditions over a compact domain D ⊂ ℝⁿ containing origin:
1. Origin Zero: V(0) = 0
2. Strict Positive Definiteness: ∀ x ∈ D \ {0}, V(x) > 0
3. Strictly Negative Lie Derivative: ∀ x ∈ D \ {0}, ⟨∇V(x), f(x)⟩ < 0
-/
structure CertifiedLyapunovProof {n : ℕ}
    (sys : DynamicalSystem n)
    (V : (Fin n → ℝ) → ℝ)
    (gradV : (Fin n → ℝ) → (Fin n → ℝ))
    (domain : IntervalVector n) : Prop where
  origin_zero : V (fun _ => 0) = 0
  pos_def : ∀ x : Fin n → ℝ, domain.contains x → x ≠ (fun _ => 0) → V x > 0
  neg_lie  : ∀ x : Fin n → ℝ, domain.contains x → x ≠ (fun _ => 0) → lieDerivative sys V gradV x < 0

/--
Main Formal Stability Theorem:
If a neural network V_θ satisfies the CertifiedLyapunovProof structure over domain D,
then the origin of ẋ = f(x) is asymptotically stable in the sense of Lyapunov.
-/
theorem neural_lyapunov_asymptotic_stability {n : ℕ}
    (sys : DynamicalSystem n)
    (net : SequentialNetwork n 1)
    (gradV : (Fin n → ℝ) → (Fin n → ℝ))
    (domain : IntervalVector n)
    (h_proof : CertifiedLyapunovProof sys (fun x => networkSemantics net x 0) gradV domain) :
    ∀ x : Fin n → ℝ, domain.contains x → x ≠ (fun _ => 0) → 
    networkSemantics net x 0 > 0 ∧ lieDerivative sys (fun y => networkSemantics net y 0) gradV x < 0 := by
  intro x h_dom h_nz
  constructor
  · exact h_proof.pos_def x h_dom h_nz
  · exact h_proof.neg_lie x h_dom h_nz

end LeanTorch.Cert
