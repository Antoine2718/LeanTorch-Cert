import LeanTorch.Core
import LeanTorch.Elab
import LeanTorch.FFI
open LeanTorch.Core LeanTorch.FFI

/-- A tiny hand-written network 2 → 2 → 1 over `Float`. -/
def demoNet : SequentialNetwork Float 2 1 :=
  ⟨.consLinear ⟨fun (i : Fin 2) (j : Fin 2) => #[#[1.0, -1.0], #[0.5, 2.0]][i.val]![j.val]!, fun (i : Fin 2) => #[0.0, -1.0][i.val]!⟩
    (.consReLU {} (.consLinear ⟨fun _ (j : Fin 2) => #[1.0, 1.0][j.val]!, fun _ => 0.5⟩ (.nil 1)))⟩

def main : IO UInt32 := do
  IO.println "== LeanTorch: shape-checked tensors with a native backend =="
  IO.println s!"backend: {backendName ()}"
  -- 1. pure-Lean forward pass (Float simulation)
  let x : Tensor.Vector 2 Float := fun i => if i.val == 0 then 2.0 else 1.0
  let y := demoNet.forward x 0
  IO.println s!"[Lean ] demoNet(2, 1) = {y}   (expected 3.5)"
  -- 2. native matmul round trip
  let m ← NativeTensor.create2D #[1, 2, 3, 4] 2 2
  let v ← NativeTensor.create1D #[2, 1]
  let r ← (← NativeTensor.matmul m v).toArray
  IO.println s!"[Native] [[1,2],[3,4]] · [2,1] = {r}   (expected #[4, 10])"
  -- 3. runtime shape errors are values, not crashes
  let bad ← NativeTensor.create1D #[1, 2, 3]
  match ← (NativeTensor.matmul m bad).toBaseIO with
  | .ok _ => IO.println "ERROR: mismatch was not detected"; return 1
  | .error e => IO.println s!"[Native] shape mismatch correctly rejected: {e}"
  if (y - 3.5).abs < 1e-12 && r == #[4.0, 10.0] then
    IO.println "OK"; return 0
  IO.println "FAILED"; return 1
