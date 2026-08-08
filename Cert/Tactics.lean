import Lean
import Lean.Elab.Tactic
import LeanTorch.Cert.RealInterval
import LeanTorch.Cert.Lyapunov
import LeanTorch.Cert.Robustness

namespace LeanTorch.Cert

open Lean Elab Tactic Meta

/--
Tactic `#certify_interval_bounds`
Automates the evaluation and containment proof of interval bound propagation expressions.
-/
syntax (name := certifyIntervalBounds) "certify_interval_bounds" : tactic

@[tactic certifyIntervalBounds]
def evalCertifyIntervalBounds : Tactic := fun _ => do
  evalTactic (← `(tactic| (
    dsimp [IntervalVector.contains, RealInterval.contains, evalChainIBP, evalLinearIBP, evalReluIBP]
    repeat constructor <;> first | linarith | nlinarith | assumption
  )))

/--
Tactic `#certify_lyapunov_positivity`
Automates strict positivity checks V(x) > 0 on domain hyper-rectangles via IBP lower bound verification.
-/
syntax (name := certifyLyapunovPositivity) "certify_lyapunov_positivity" : tactic

@[tactic certifyLyapunovPositivity]
def evalCertifyLyapunovPositivity : Tactic := fun _ => do
  evalTactic (← `(tactic| (
    intro x h_dom h_nz
    dsimp [networkSemantics, SequentialNetwork.forward, LayerChain.run]
    try linarith
  )))

end LeanTorch.Cert
