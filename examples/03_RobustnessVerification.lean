import LeanTorch

open LeanTorch.Core
open LeanTorch.Cert

namespace Examples.RobustnessVerification

/--
Example 3: Verifying L_infinity local adversarial robustness for a 1D -> 1D neural network
using Interval Bound Propagation (IBP).
-/

-- Define a single-layer neural network with weight W = 2.0, bias b = -1.0
def simpleNet : SequentialNetwork 1 1 := {
  chain := LayerChain.consLinear
    { weights := fun _ => 2.0, bias := fun _ => -1.0 }
    (LayerChain.nil 1)
}

-- Input reference x0 = 1.0, perturbation radius ε = 0.1
def x0 : Fin 1 → ℝ := fun _ => 1.0
def epsilon : ℝ := 0.1
have h_eps : 0 ≤ epsilon := by linarith

-- Perturbation input interval B_ε(x0) = [0.9, 1.1]
def inputPerturbedBox : IntervalVector 1 :=
  lInfBall x0 epsilon h_eps

-- Expected Output Bounds [y_min, y_max] = [0.8, 1.2]
def y_min : Fin 1 → ℝ := fun _ => 0.8
def y_max : Fin 1 → ℝ := fun _ => 1.2

/--
Theorem: The network output is strictly guaranteed to lie within [0.8, 1.2]
for all inputs in [0.9, 1.1].
-/
theorem simpleNet_is_robust :
    IsLocallyRobust simpleNet x0 epsilon h_eps y_min y_max := by
  intro x hx i
  dsimp [networkSemantics, SequentialNetwork.forward, LayerChain.run, DenseLayer.forward, Tensor.add, Tensor.vecMatMul]
  have h_in := hx i
  dsimp [lInfBall, x0, epsilon] at h_in
  constructor
  · linarith [h_in.1]
  · linarith [h_in.2]

end Examples.RobustnessVerification
