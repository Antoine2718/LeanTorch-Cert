import Lean
import LeanTorch.Core.Network
namespace LeanTorch.Elab
open Lean Elab Command

structure RationalJSON where
  numerator   : Int
  denominator : Nat
  deriving FromJson, ToJson

structure TensorJSON where
  shape  : List Nat
  values : List RationalJSON
  deriving FromJson, ToJson

structure LayerJSON where
  name           : String
  layer_type     : String
  in_shape       : List Nat
  out_shape      : List Nat
  weights        : Option TensorJSON := none
  bias           : Option TensorJSON := none
  activation     : String := "Identity"
  negative_slope : Option RationalJSON := none
  deriving FromJson, ToJson

structure ModelJSON where
  model_name   : String
  input_shape  : List Nat
  output_shape : List Nat
  layers       : List LayerJSON
  deriving FromJson, ToJson

/-- Flattened operations of a network. -/
inductive Op where
  | dense (inD outD : Nat) (w b : Array RationalJSON)
  | relu (d : Nat)
  | leaky (d : Nat) (slope : RationalJSON)
  | tanh (d : Nat)
  deriving Inhabited

private def lastDim (s : List Nat) : Option Nat := s.getLast?

/-- Turn the layer list into primitive operations (fusing `activation` fields). -/
def toOps (spec : ModelJSON) : Except String (Array Op) := do
  let mut ops : Array Op := #[]
  for l in spec.layers do
    let mut width ← (lastDim l.out_shape).elim (.error s!"layer {l.name}: empty out_shape") .ok
    if l.layer_type == "Linear" then
      let some w := l.weights | throw s!"layer {l.name}: Linear layer without weights"
      let [o, i] := w.shape | throw s!"layer {l.name}: weights must have shape [out, in]"
      unless w.values.length == o * i do throw s!"layer {l.name}: weights size mismatch"
      let b : Array RationalJSON ← match l.bias with
        | none => pure (Array.mkArray o ⟨0, 1⟩)
        | some bt =>
          unless bt.values.length == o do throw s!"layer {l.name}: bias size mismatch"
          pure bt.values.toArray
      ops := ops.push (.dense i o w.values.toArray b)
      width := o
    else if l.layer_type != "Activation" then
      throw s!"layer {l.name}: unsupported layer type `{l.layer_type}`"
    match l.activation with
    | "Identity" => pure ()
    | "ReLU"      => ops := ops.push (.relu width)
    | "Tanh"      => ops := ops.push (.tanh width)
    | "LeakyReLU" => ops := ops.push (.leaky width (l.negative_slope.getD ⟨1, 100⟩))
    | a => throw s!"layer {l.name}: activation `{a}` is not supported by LeanTorch (use ReLU, LeakyReLU, Tanh)"
  return ops

def opIn : Op → Nat
  | .dense i _ _ _ => i | .relu d => d | .leaky d _ => d | .tanh d => d
def opOut : Op → Nat
  | .dense _ o _ _ => o | .relu d => d | .leaky d _ => d | .tanh d => d

def ratTerm (α : TSyntax `term) (r : RationalJSON) : MacroM (TSyntax `term) := do
  let n := Syntax.mkNumLit (toString r.numerator.natAbs)
  let d := Syntax.mkNumLit (toString r.denominator)
  let pos : TSyntax `term ← if r.denominator == 1 then `(($n : $α)) else `((($n : $α) / ($d : $α)))
  if r.numerator < 0 then `(-$pos) else pure pos

def arrTerm (α : TSyntax `term) (xs : Array RationalJSON) : MacroM (TSyntax `term) := do
  let ts ← xs.mapM (ratTerm α)
  `((#[$ts,*] : Array $α))

def opTerm (α : TSyntax `term) (op : Op) (rest : TSyntax `term) : MacroM (TSyntax `term) := do
  match op with
  | .dense i o w b =>
    let iS := Syntax.mkNumLit (toString i)
    let oS := Syntax.mkNumLit (toString o)
    let wA ← arrTerm α w
    let bA ← arrTerm α b
    `(LeanTorch.Core.LayerChain.consLinear
        (⟨fun (r : Fin $oS) (c : Fin $iS) => ($wA).getD (r.val * $iS + c.val) 0,
          fun (r : Fin $oS) => ($bA).getD r.val 0⟩ : LeanTorch.Core.DenseLayer $α $iS $oS) $rest)
  | .relu d =>
    `(LeanTorch.Core.LayerChain.consReLU (d := $(Syntax.mkNumLit (toString d))) {} $rest)
  | .tanh d =>
    `(LeanTorch.Core.LayerChain.consTanh (d := $(Syntax.mkNumLit (toString d))) {} $rest)
  | .leaky d s =>
    let sT ← ratTerm α s
    `(LeanTorch.Core.LayerChain.consLeakyReLU (d := $(Syntax.mkNumLit (toString d))) { slope := $sT } $rest)

/-- Shared implementation of both import commands. -/
def importModel (path : TSyntax `str) (id : TSyntax `ident) (α : TSyntax `term)
    (noncomp : Bool) : CommandElabM Unit := do
  let file := path.getString
  unless (← System.FilePath.pathExists file) do
    throwError "[LeanTorch] model file not found: {file} (path is relative to where `lake build` runs)"
  let content ← IO.FS.readFile file
  let json ← match Json.parse content with
    | .ok j => pure j
    | .error e => throwError "[LeanTorch] invalid JSON in {file}: {e}"
  let spec ← match fromJson? (α := ModelJSON) json with
    | .ok s => pure s
    | .error e => throwError "[LeanTorch] unexpected schema in {file}: {e}"
  let ops ← match toOps spec with
    | .ok o => pure o
    | .error e => throwError "[LeanTorch] {e}"
  if ops.isEmpty then throwError "[LeanTorch] model {spec.model_name} has no supported layers"
  for k in [1:ops.size] do
    if opOut ops[k-1]! != opIn ops[k]! then
      throwError "[LeanTorch] width mismatch between op {k-1} (out {opOut ops[k-1]!}) and op {k} (in {opIn ops[k]!})"
  let inD := Syntax.mkNumLit (toString (opIn ops[0]!))
  let outD := Syntax.mkNumLit (toString (opOut ops[ops.size - 1]!))
  let mut chain : TSyntax `term ← `(LeanTorch.Core.LayerChain.nil _)
  for op in ops.reverse do
    chain ← liftMacroM (opTerm α op chain)
  let cmd ← if noncomp then
      `(noncomputable def $id : LeanTorch.Core.SequentialNetwork $α $inD $outD := ⟨$chain⟩)
    else
      `(def $id : LeanTorch.Core.SequentialNetwork $α $inD $outD := ⟨$chain⟩)
  logInfo m!"[LeanTorch] imported '{spec.model_name}': {ops.size} ops, {inD} → {outD}"
  elabCommand cmd

/-- `#import_torch_model "m.json" as Net : Float` (computable scalar type). -/
syntax "#import_torch_model " str " as " ident " : " term : command
/-- `#import_torch_model_noncomputable "m.json" as Net : ℝ` (for `ℝ`). -/
syntax "#import_torch_model_noncomputable " str " as " ident " : " term : command

elab_rules : command
  | `(#import_torch_model $path:str as $id:ident : $α:term) => importModel path id α false
  | `(#import_torch_model_noncomputable $path:str as $id:ident : $α:term) => importModel path id α true

end LeanTorch.Elab
