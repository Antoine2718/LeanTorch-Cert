import LeanTorch
/-!
# Example 4 — local L∞ robustness of `y = 2x − 1` on `[0.9, 1.1]`
The output range is `[0.8, 1.2]`.  The proof is by hand; for general networks use
`LeanTorch.Cert.verify_robustness_via_ibp` (soundness of interval bound propagation) together
with the exact rational IBP in `python/leantorch_exporter/ibp.py`.
-/
open LeanTorch.Core LeanTorch.Cert

namespace Examples.Robustness

def simpleNet : SequentialNetwork ℝ 1 1 :=
  ⟨.consLinear ⟨fun _ _ => 2, fun _ => -1⟩ (.nil 1)⟩

def x0 : Fin 1 → ℝ := fun _ => 1
def ε : ℝ := 1 / 10
theorem hε : 0 ≤ ε := by unfold ε; norm_num

def ymin : Fin 1 → ℝ := fun _ => 4 / 5
def ymax : Fin 1 → ℝ := fun _ => 6 / 5

theorem simpleNet_is_robust : IsLocallyRobust simpleNet x0 ε hε ymin ymax := by
  intro x hx k
  have hk : k = 0 := Subsingleton.elim _ _
  subst hk
  have h := hx 0
  have h1 : (1 : ℝ) - 1 / 10 ≤ x 0 := h.1
  have h2 : x 0 ≤ (1 : ℝ) + 1 / 10 := h.2
  show 4 / 5 ≤ 0 + 2 * x 0 + -1 ∧ 0 + 2 * x 0 + -1 ≤ 6 / 5
  constructor <;> linarith

end Examples.Robustness
