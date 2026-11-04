import LeanTorch.Core
import LeanTorch.Elab
/-!
# Example 1 — shape errors are compile errors
Uncomment the last definition to see Lean reject a width mismatch.
-/
open LeanTorch.Core

namespace Examples.CompileTimeShapes

def myNetwork : SequentialNetwork Float 2 1 :=
  ⟨torch_net % [linear (2, 8), relu (8), linear (8, 4), tanh (4), linear (4, 1)]⟩

#assert_network_shape myNetwork : 2 => 1

-- def broken : SequentialNetwork Float 2 1 :=
--   ⟨torch_net % [linear (2, 8), relu (7), linear (8, 1)]⟩   -- ✗ type error: 8 ≠ 7

end Examples.CompileTimeShapes
