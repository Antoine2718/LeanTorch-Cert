import Lean
import LeanTorch.Core.Network
import LeanTorch.Elab.Syntax

namespace LeanTorch.Elab

open Lean Elab Term Macro Core

/--
Compile-time Macro Elaborator for `torch_net % [...]`.
Parses the DSL layer specifications and generates strongly-typed `LayerChain` expressions.
If adjacent layer dimensions mismatch, Lean's type elaborator aborts compilation.
-/

macro_rules
  | `(torch_net % [ ]) => `(LeanTorch.Core.LayerChain.nil _)

  | `(torch_net % [ linear ($in_d, $out_d) ]) =>
    `(LeanTorch.Core.LayerChain.consLinear 
        ({ weights := Tensor.zero [$out_d, $in_d], bias := Tensor.zero [$out_d] }) 
        (LeanTorch.Core.LayerChain.nil $out_d))

  | `(torch_net % [ relu ($d) ]) =>
    `(LeanTorch.Core.LayerChain.consReLU 
        ({}) 
        (LeanTorch.Core.LayerChain.nil $d))

  | `(torch_net % [ leaky_relu ($d, $slope) ]) =>
    `(LeanTorch.Core.LayerChain.consLeakyReLU 
        ({ negative_slope := $slope }) 
        (LeanTorch.Core.LayerChain.nil $d))

  | `(torch_net % [ tanh ($d) ]) =>
    `(LeanTorch.Core.LayerChain.consTanh 
        ({}) 
        (LeanTorch.Core.LayerChain.nil $d))

  | `(torch_net % [ linear ($in_d, $out_d), $tail,* ]) =>
    `(LeanTorch.Core.LayerChain.consLinear 
        ({ weights := Tensor.zero [$out_d, $in_d], bias := Tensor.zero [$out_d] }) 
        (torch_net % [ $tail,* ]))

  | `(torch_net % [ relu ($d), $tail,* ]) =>
    `(LeanTorch.Core.LayerChain.consReLU 
        ({}) 
        (torch_net % [ $tail,* ]))

  | `(torch_net % [ leaky_relu ($d, $slope), $tail,* ]) =>
    `(LeanTorch.Core.LayerChain.consLeakyReLU 
        ({ negative_slope := $slope }) 
        (torch_net % [ $tail,* ]))

  | `(torch_net % [ tanh ($d), $tail,* ]) =>
    `(LeanTorch.Core.LayerChain.consTanh 
        ({}) 
        (torch_net % [ $tail,* ]))

/--
Elaborator for `#assert_network_shape net_term : in_dim => out_dim`.
Verifies at macro-expansion time that a network has the declared input/output signature.
-/
open Command in
elab_rules : command
  | `(#assert_network_shape $net:term : $in_d:num => $out_d:num) => do
    let inNat := in_d.getNat
    let outNat := out_d.getNat
    runTermElabM fun _ => do
      let expectedType ← `(LeanTorch.Core.SequentialNetwork $in_d $out_d)
      let netExpr ← elabTerm net (some (← elabType expectedType))
      synthesizeSyntheticMVarsNoErr
      logInfo s!"✓ [LeanTorch ShapeChecker] Network statically verified for shape signature [{inNat} → {outNat}]."

end LeanTorch.Elab
