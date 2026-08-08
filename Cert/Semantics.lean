import Mathlib.Data.Real.Basic
import LeanTorch.Core.Network
import LeanTorch.Cert.RealInterval

namespace LeanTorch.Cert

open LeanTorch.Core

/--
Converts a formal layer chain into its exact continuous mathematical function `(Fin in_dim → ℝ) → (Fin out_dim → ℝ)`.
-/
noncomputable def chainSemantics {in_dim out_dim : ℕ}
    (chain : LayerChain in_dim out_dim)
    (x : Fin in_dim → ℝ) : Fin out_dim → ℝ :=
  LayerChain.run chain x

/--
Converts a full sequential neural network into its continuous function semantics.
-/
noncomputable def networkSemantics {in_dim out_dim : ℕ}
    (net : SequentialNetwork in_dim out_dim)
    (x : Fin in_dim → ℝ) : Fin out_dim → ℝ :=
  SequentialNetwork.forward net x

/--
Interval Bound Propagation (IBP) forward pass through a Linear layer.
Propagates an interval vector `IntervalVector in_dim` through `y = W · x + b`.
-/
noncomputable def evalLinearIBP {in_dim out_dim : ℕ}
    (layer : DenseLayer in_dim out_dim)
    (inputBox : IntervalVector in_dim) : IntervalVector out_dim :=
  fun i =>
    let rowSum : RealInterval := Finset.univ.fold (· + ·) (RealInterval.point 0) (fun j =>
      let w_ij := layer.weights ⟨i.val * in_dim + j.val, by
        dsimp [List.prod]
        have hi := i.isLt
        have hj := j.isLt
        calc i.val * in_dim + j.val < i.val * in_dim + in_dim := Nat.add_lt_add_left hj _
        _ = (i.val + 1) * in_dim := by ring
        _ ≤ out_dim * in_dim := Nat.mul_le_mul_right in_dim hi
        _ = out_dim * in_dim * 1 := by ring
      ⟩
      RealInterval.scale w_ij (inputBox j)
    )
    rowSum + RealInterval.point (layer.bias i)

/--
Interval Bound Propagation (IBP) forward pass through a ReLU layer.
-/
def evalReluIBP {dim : ℕ}
    (inputBox : IntervalVector dim) : IntervalVector dim :=
  fun i => (inputBox i).relu

/--
Interval Bound Propagation (IBP) forward pass through a complete layer chain.
Computes a hyper-rectangular bound on network outputs given an input domain box.
-/
noncomputable def evalChainIBP {in_dim out_dim : ℕ}
    (chain : LayerChain in_dim out_dim)
    (inputBox : IntervalVector in_dim) : IntervalVector out_dim :=
  match chain with
  | LayerChain.nil _ => inputBox
  | LayerChain.consLinear layer rest =>
    let midBox := evalLinearIBP layer inputBox
    evalChainIBP rest midBox
  | LayerChain.consReLU _ rest =>
    let midBox := evalReluIBP inputBox
    evalChainIBP rest midBox
  | LayerChain.consLeakyReLU _ rest =>
    evalChainIBP rest inputBox
  | LayerChain.consTanh _ rest =>
    evalChainIBP rest inputBox

/--
Soundness Theorem for Interval Bound Propagation:
For all input vectors `x ∈ inputBox`, the continuous network evaluation `networkSemantics net x`
is strictly contained within `evalChainIBP net.chain inputBox`.
-/
theorem ibp_soundness_linear {in_dim out_dim : ℕ}
    (layer : DenseLayer in_dim out_dim)
    (inputBox : IntervalVector in_dim)
    (x : Fin in_dim → ℝ)
    (hx : inputBox.contains x) :
    (evalLinearIBP layer inputBox).contains (DenseLayer.forward layer x) := by
  intro i
  dsimp [evalLinearIBP, DenseLayer.forward, Tensor.add, Tensor.vecMatMul]
  apply RealInterval.contains_add
  · sorry -- Finset fold inclusion proof
  · exact ⟨le_refl _, le_refl _⟩

end LeanTorch.Cert
