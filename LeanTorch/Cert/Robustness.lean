import LeanTorch.Cert.Semantics
namespace LeanTorch.Cert
noncomputable section
open LeanTorch.Core

/-- `L∞` ball `{x | ∀ i, |x i - x0 i| ≤ ε}` as a box. -/
def lInfBall {n : Nat} (x0 : Fin n → ℝ) (ε : ℝ) (hε : 0 ≤ ε) : IntervalVector n :=
  fun i => ⟨x0 i - ε, x0 i + ε, le_trans (sub_le_self _ hε) (le_add_of_nonneg_right hε)⟩

/-- Local robustness: every input in the `ε`-ball maps into `[y_min, y_max]`. -/
def IsLocallyRobust {i o : Nat} (net : SequentialNetwork ℝ i o) (x0 : Fin i → ℝ) (ε : ℝ)
    (hε : 0 ≤ ε) (y_min y_max : Fin o → ℝ) : Prop :=
  ∀ x : Fin i → ℝ, (lInfBall x0 ε hε).contains x →
    ∀ k : Fin o, y_min k ≤ networkSemantics net x k ∧ networkSemantics net x k ≤ y_max k

/-- **Certificate checker**: if IBP on the ball returns a box `out` inside `[y_min, y_max]`,
the network is locally robust.  No `sorry`: this follows from `evalChainIBP_sound`. -/
theorem verify_robustness_via_ibp {i o : Nat} (net : SequentialNetwork ℝ i o)
    (x0 : Fin i → ℝ) (ε : ℝ) (hε : 0 ≤ ε) (y_min y_max : Fin o → ℝ)
    (out : IntervalVector o)
    (h : evalChainIBP net.chain (lInfBall x0 ε hε) = some out)
    (hb : ∀ k, y_min k ≤ (out k).lo ∧ (out k).hi ≤ y_max k) :
    IsLocallyRobust net x0 ε hε y_min y_max := by
  intro x hx k
  have hc := evalChainIBP_sound net.chain _ out h x hx k
  exact ⟨le_trans (hb k).1 hc.1, le_trans hc.2 (hb k).2⟩

end -- noncomputable section
end LeanTorch.Cert
