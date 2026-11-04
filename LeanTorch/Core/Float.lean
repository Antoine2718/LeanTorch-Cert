import LeanTorch.Core.Basic
namespace LeanTorch.Core

/-- Executable activations over IEEE doubles (for simulation / testing only:
floating-point results are *not* covered by the real-number certificates). -/
instance : Activations Float where
  relu x        := if x > 0 then x else 0
  leakyRelu s x := if x ≥ 0 then x else s * x
  tanh          := Float.tanh

end LeanTorch.Core
