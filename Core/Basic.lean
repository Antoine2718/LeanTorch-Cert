import Mathlib.Data.Real.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.List.Prod

namespace LeanTorch.Core

/--
Helper utilities and index conversion logic for shape-indexed multi-dimensional tensors.
Provides bijectivity mapping between flat 1D linear memory addresses and multi-dimensional coordinate vectors.
-/

/--
Calculates the strides for a given shape list [d_0, d_1, ..., d_{k-1}].
For example, for shape [2, 3, 4], strides are [12, 4, 1].
-/
def computeStrides : List ℕ → List ℕ
  | [] => []
  | [d] => [1]
  | d :: ds =>
    let tailStrides := computeStrides ds
    let nextStride := ds.prod
    nextStride :: tailStrides

/--
Converts a multi-dimensional coordinate vector into a 1D flat index.
-/
def multiToFlat (shape : List ℕ) (coords : List ℕ) : ℕ :=
  let strides := computeStrides shape
  (List.zip strides coords).foldl (fun acc (s, c) => acc + s * c) 0

/--
Converts a 1D flat index back into a multi-dimensional coordinate vector.
-/
def flatToMulti : List ℕ → ℕ → List ℕ
  | [], _ => []
  | d :: ds, idx =>
    let stride := ds.prod
    if stride = 0 then
      0 :: flatToMulti ds 0
    else
      let coord := idx / stride
      let rem := idx % stride
      coord :: flatToMulti ds rem

/--
Smooth activation helper functions defined on ℝ.
-/
noncomputable def realReLU (x : ℝ) : ℝ :=
  max 0 x

noncomputable def realLeakyReLU (slope : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then x else slope * x

noncomputable def realSigmoid (x : ℝ) : ℝ :=
  1 / (1 + Real.exp (-x))

noncomputable def realTanh (x : ℝ) : ℝ :=
  (Real.exp x - Real.exp (-x)) / (Real.exp x + Real.exp (-x))

end LeanTorch.Core
