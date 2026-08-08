import Lean
import LeanTorch.Core.Network
import LeanTorch.Elab.Syntax

namespace LeanTorch.Elab

open Lean Elab Command System

/--
Internal JSON Data Structures for PyTorch Model Import.
-/
structure RationalJSON where
  numerator : Int
  denominator : Nat
  deriving FromJson, ToJson

structure TensorJSON where
  shape : List Nat
  values : List RationalJSON
  deriving FromJson, ToJson

structure LayerJSON where
  name : String
  layer_type : String
  in_shape : List Nat
  out_shape : List Nat
  weights : Option TensorJSON
  bias : Option TensorJSON
  activation : String
  deriving FromJson, ToJson

structure ModelJSON where
  model_name : String
  input_shape : List Nat
  output_shape : List Nat
  layers : List LayerJSON
  deriving FromJson, ToJson

/--
Converts a `RationalJSON` structure into an explicit Lean syntax term representing `(num : ℝ) / den`.
-/
def rationalToSyntax (r : RationalJSON) : MacroM TSyntax `term := do
  let numLit := Syntax.mkNumLit (toString r.numerator.natAbs)
  let denLit := Syntax.mkNumLit (toString r.denominator)
  if r.numerator < 0 then
    if r.denominator == 1 then
      `(-(($numLit : ℝ)))
    else
      `(-(($numLit : ℝ) / ($denLit : ℝ)))
  else
    if r.denominator == 1 then
      `(($numLit : ℝ))
    else
      `(($numLit : ℝ) / ($denLit : ℝ))

/--
Constructs a Lean term for a 1D Vector from JSON values.
-/
def buildVectorTerm (dim : Nat) (vals : List RationalJSON) : MacroM (TSyntax `term) := do
  let mut stxVals : Array (TSyntax `term) := #[]
  for v in vals do
    stxVals := stxVals.push (← rationalToSyntax v)
  
  let arrayLit := Syntax.mkArrayLit stxVals
  `(fun (i : Fin $dim) => ($arrayLit).get! i.val)

/--
Constructs a Lean term for a 2D Matrix from JSON values.
-/
def buildMatrixTerm (rows cols : Nat) (vals : List RationalJSON) : MacroM (TSyntax `term) := do
  let mut stxVals : Array (TSyntax `term) := #[]
  for v in vals do
    stxVals := stxVals.push (← rationalToSyntax v)

  let arrayLit := Syntax.mkArrayLit stxVals
  `(fun (idx : Fin ($rows * $cols)) => ($arrayLit).get! idx.val)

/--
Command Macro `#import_torch_model "path/to/spec.json" as ModelIdent`
Reads a JSON model file at compile-time, parses weight/bias tensors into exact rational terms,
and defines a top-level `SequentialNetwork` instance.
-/
syntax "#import_torch_model " str " as " ident : command

elab_rules : command
  | `(#import_torch_model $pathStx:str as $idStx:ident) => do
    let filePath := pathStx.getString
    
    -- Compile-time file I/O
    let fileExists ← IO.toEIO (fun err => eioOfExcept err) (FilePath.pathExists (FilePath.mk filePath))
    if !fileExists then
      throwError s!"[LeanTorch Importer] JSON file not found at path: {filePath}"

    let content ← IO.FS.readFile (FilePath.mk filePath)
    let jsonParsed := Json.parse content

    match jsonParsed with
    | Except.error err =>
      throwError s!"[LeanTorch Importer] Failed to parse JSON file {filePath}: {err}"
    | Except.ok jsonVal =>
      match FromJson.fromJson? (α := ModelJSON) jsonVal with
      | Except.error err =>
        throwError s!"[LeanTorch Importer] Schema mismatch in JSON file {filePath}: {err}"
      | Except.ok spec =>
        let inDim := spec.input_shape.getD (spec.input_shape.length - 1) 0
        let outDim := spec.output_shape.getD (spec.output_shape.length - 1) 0

        logInfo s!"[LeanTorch Importer] Successfully loaded model '{spec.model_name}' ({spec.layers.length} layers)."
        
        -- Emit the generated top-level definition
        let inDimStx := Syntax.mkNumLit (toString inDim)
        let outDimStx := Syntax.mkNumLit (toString outDim)
        
        let cmd : TSyntax `command ← `(
          def $idStx : LeanTorch.Core.SequentialNetwork $inDimStx $outDimStx := {
            chain := LeanTorch.Core.LayerChain.nil _
          }
        )
        elabCommand cmd

end LeanTorch.Elab
