import LeanTorch

open LeanTorch.Core
open LeanTorch.Elab

namespace Examples.CompileTimeShapes

/--
Example 1: Demonstrates compile-time shape verification using Lean 4 macros and types.
Any mismatch in dimensions between adjacent layers causes a Lean compilation error.
-/

-- Declare a 3-layer sequential network using the custom `torch_net % [...]` DSL
def myNetwork : SequentialNetwork 2 1 := {
  chain := torch_net % [
    linear (2, 8),
    relu (8),
    linear (8, 4),
    tanh (4),
    linear (4, 1)
  ]
}

-- Validate network shape statically at compile time
#assert_network_shape myNetwork : 2 => 1

-- Forward pass evaluation over a sample input vector
def sampleInput : Tensor.Vector 2 ℝ := fun i => match i.val with
  | 0 => 1.0
  | _ => -0.5

noncomputable def sampleOutput : Tensor.Vector 1 ℝ :=
  myNetwork.forward sampleInput

end Examples.CompileTimeShapes
