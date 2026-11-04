import Lean
import LeanTorch.Core.Network
import LeanTorch.Elab.Syntax
namespace LeanTorch.Elab
open Lean

private def expandLayer (l : TSyntax `torch_layer) (rest : TSyntax `term) :
    MacroM (TSyntax `term) := do
  match l with
  | `(torch_layer| $id:ident ($args,*)) =>
    let a := args.getElems
    match id.getId.toString, a.size with
    | "linear", 2 =>
      `(LeanTorch.Core.LayerChain.consLinear (LeanTorch.Core.DenseLayer.zero $(a[0]!) $(a[1]!)) $rest)
    | "relu", 1 =>
      `(LeanTorch.Core.LayerChain.consReLU (d := $(a[0]!)) {} $rest)
    | "tanh", 1 =>
      `(LeanTorch.Core.LayerChain.consTanh (d := $(a[0]!)) {} $rest)
    | "leaky_relu", 2 =>
      `(LeanTorch.Core.LayerChain.consLeakyReLU (d := $(a[0]!)) { slope := $(a[1]!) } $rest)
    | name, n =>
      Macro.throwErrorAt l s!"torch_net: unknown layer or wrong arity `{name}/{n}` \
        (expected linear/2, relu/1, tanh/1, leaky_relu/2)"
  | _ => Macro.throwUnsupported

macro_rules
  | `(torch_net % [ $layers,* ]) => do
    let mut acc : TSyntax `term ← `(LeanTorch.Core.LayerChain.nil _)
    for l in layers.getElems.reverse do
      acc ← expandLayer l acc
    return acc

macro_rules
  | `(#assert_network_shape $n : $i => $o) =>
    `(#check ($n : LeanTorch.Core.SequentialNetwork _ $i $o))

end LeanTorch.Elab
