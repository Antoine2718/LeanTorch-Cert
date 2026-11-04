import Mathlib.Data.Real.Basic
import LeanTorch.Core.Basic
namespace LeanTorch.Cert
noncomputable section
open LeanTorch.Core

/-- LeakyReLU on ℝ. -/
noncomputable def leakyFn (s x : ℝ) : ℝ := if 0 ≤ x then x else s * x

/-- Exact real-number semantics of the activations (what the certificates talk about). -/
noncomputable instance : Activations ℝ where
  relu x        := max 0 x
  leakyRelu s x := leakyFn s x
  tanh x        := (Real.exp x - Real.exp (-x)) / (Real.exp x + Real.exp (-x))

end -- noncomputable section
end LeanTorch.Cert
