import Lean
/-!
Syntax of the `torch_net % [...]` DSL and of the `#assert_network_shape` command.
Layer names are parsed as plain identifiers (not keywords), so `relu`, `tanh`, `linear`
remain usable as ordinary names everywhere else.
-/
namespace LeanTorch.Elab
open Lean

declare_syntax_cat torch_layer
syntax ident "(" term,* ")" : torch_layer

/-- `torch_net % [linear (2, 8), relu (8), linear (8, 1)]` builds a zero-initialised
`LayerChain`.  Adjacent widths are checked by Lean's type checker. -/
syntax "torch_net" "%" "[" torch_layer,* "]" : term

/-- `#assert_network_shape net : 2 => 1` fails to compile unless `net : SequentialNetwork _ 2 1`. -/
syntax "#assert_network_shape " term " : " num " => " num : command

end LeanTorch.Elab
