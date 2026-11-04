import LeanTorch.Core.Network
import LeanTorch.Cert.RealInterval
namespace LeanTorch.Cert
noncomputable section
open LeanTorch.Core

/-- Real-number semantics of a network. -/
noncomputable def networkSemantics {i o : Nat} (net : SequentialNetwork ℝ i o) (x : Fin i → ℝ) :
    Fin o → ℝ := net.forward x

/-! ### Interval sums -/

theorem contains_sumFin : ∀ (n : Nat) (I : Fin n → RealInterval) (f : Fin n → ℝ),
    (∀ j, (I j).contains (f j)) → (sumFin n I).contains (sumFin n f)
  | 0, _, _, _ => ⟨le_refl _, le_refl _⟩
  | n + 1, I, f, h =>
    RealInterval.contains_add
      (contains_sumFin n (fun j => I j.castSucc) (fun j => f j.castSucc) (fun j => h j.castSucc))
      (h (Fin.last n))

/-! ### Interval Bound Propagation, layer by layer -/

noncomputable def linearIBP {i m : Nat} (l : DenseLayer ℝ i m) (B : IntervalVector i) :
    IntervalVector m :=
  fun r => sumFin i (fun j => RealInterval.scale (l.weights r j) (B j)) + RealInterval.point (l.bias r)

theorem linearIBP_sound {i m : Nat} (l : DenseLayer ℝ i m) (B : IntervalVector i)
    (x : Fin i → ℝ) (hx : B.contains x) : (linearIBP l B).contains (l.forward x) :=
  fun r =>
    RealInterval.contains_add
      (contains_sumFin i _ _ (fun j => RealInterval.contains_scale (hx j)))
      ⟨le_refl _, le_refl _⟩

def reluIBP {d : Nat} (B : IntervalVector d) : IntervalVector d := fun k => (B k).relu

theorem reluIBP_sound {d : Nat} (B : IntervalVector d) (x : Fin d → ℝ) (hx : B.contains x) :
    (reluIBP B).contains (fun k => Activations.relu (x k)) :=
  fun k => RealInterval.contains_relu (hx k)

def leakyIBP {d : Nat} (s : ℝ) (hs : 0 ≤ s) (B : IntervalVector d) : IntervalVector d :=
  fun k => (B k).leaky s hs

theorem leakyIBP_sound {d : Nat} (s : ℝ) (hs : 0 ≤ s) (B : IntervalVector d) (x : Fin d → ℝ)
    (hx : B.contains x) : (leakyIBP s hs B).contains (fun k => Activations.leakyRelu s (x k)) :=
  fun k => RealInterval.contains_leaky hs (hx k)

/-- IBP through a whole chain.  Returns `none` for layers that are not (yet) supported
(currently `tanh`, and LeakyReLU with a negative slope), so a `some` result is always sound. -/
noncomputable def evalChainIBP : {i o : Nat} → LayerChain ℝ i o → IntervalVector i →
    Option (IntervalVector o)
  | _, _, .nil _,                B => some B
  | _, _, .consLinear l rest,    B => evalChainIBP rest (linearIBP l B)
  | _, _, .consReLU _ rest,      B => evalChainIBP rest (reluIBP B)
  | _, _, .consLeakyReLU l rest, B =>
      if hs : 0 ≤ l.slope then evalChainIBP rest (leakyIBP l.slope hs B) else none
  | _, _, .consTanh _ _,         _ => none

/-- **Soundness of IBP**: whenever `evalChainIBP` returns a box, it contains the true
network output for every input in the input box. -/
theorem evalChainIBP_sound : ∀ {i o : Nat} (c : LayerChain ℝ i o) (B : IntervalVector i)
    (out : IntervalVector o), evalChainIBP c B = some out →
    ∀ x : Fin i → ℝ, B.contains x → out.contains (c.run x)
  | _, _, .nil _, B, out, h, x, hx => by
    have h' : some B = some out := h
    cases h'
    exact hx
  | _, _, .consLinear l rest, B, out, h, x, hx =>
    evalChainIBP_sound rest _ out h _ (linearIBP_sound l B x hx)
  | _, _, .consReLU _ rest, B, out, h, x, hx =>
    evalChainIBP_sound rest _ out h _ (reluIBP_sound B x hx)
  | _, _, .consLeakyReLU l rest, B, out, h, x, hx => by
    have h' : (if hs : 0 ≤ l.slope then evalChainIBP rest (leakyIBP l.slope hs B) else none)
        = some out := h
    by_cases hs : 0 ≤ l.slope
    · rw [dif_pos hs] at h'
      exact evalChainIBP_sound rest _ out h' _ (leakyIBP_sound l.slope hs B x hx)
    · rw [dif_neg hs] at h'
      cases h'
  | _, _, .consTanh _ _, _, out, h, _, _ => by
    have h' : (none : Option (IntervalVector _)) = some out := h
    cases h'

end -- noncomputable section
end LeanTorch.Cert
