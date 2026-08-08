import Lean
import LeanTorch.Core.Network

namespace LeanTorch.Elab

open Lean

/--
Syntax definitions for the `torch_net % [...]` domain-specific language (DSL).
Enables declarative neural network construction in native Lean 4 syntax.
-/

declare_syntax_cat torch_layer

syntax "linear" "(" num "," num ")" : torch_layer
syntax "relu" "(" num ")"          : torch_layer
syntax "leaky_relu" "(" num "," term ")" : torch_layer
syntax "tanh" "(" num ")"          : torch_layer

syntax "torch_net % " "[" torch_layer,* "]" : term

/--
Syntax command for compile-time model metadata inspection and shape assertion.
-/
syntax "#assert_network_shape " term " : " num " => " num : command

end LeanTorch.Elab
