import LeanTorch.Core
import LeanTorch.Elab
import LeanTorch.FFI
open LeanTorch.Core LeanTorch.FFI

def approx (a b : Float) : Bool := (a - b).abs < 1e-9

structure Counter where
  pass : Nat := 0
  fail : Nat := 0

def check (c : IO.Ref Counter) (name : String) (ok : Bool) : IO Unit := do
  if ok then c.modify fun s => { s with pass := s.pass + 1 }
  else
    IO.eprintln s!"  FAIL: {name}"
    c.modify fun s => { s with fail := s.fail + 1 }

def raises {α : Type} (act : IO α) : IO Bool := do
  try let _ ← act; return false catch _ => return true

/-- 2×2 / 2×1 example network used by several tests. -/
def netA : SequentialNetwork Float 2 1 :=
  ⟨.consLinear ⟨fun (i : Fin 2) (j : Fin 2) => #[#[1.0, -1.0], #[0.5, 2.0]][i.val]![j.val]!, fun (i : Fin 2) => #[0.0, -1.0][i.val]!⟩
    (.consReLU {} (.consLinear ⟨fun _ (j : Fin 2) => #[1.0, 1.0][j.val]!, fun _ => 0.5⟩ (.nil 1)))⟩

def main : IO UInt32 := do
  let c ← IO.mkRef ({} : Counter)
  -- index arithmetic: exhaustive round trip on shape [2,3,4]
  let shape := [2, 3, 4]
  let mut ok := true
  for a in [0:2] do for b in [0:3] do for d in [0:4] do
    let f := multiToFlat shape [a, b, d]
    ok := ok && flatToMulti shape f == [a, b, d] && f == a * 12 + b * 4 + d
  check c "flat/multi round trip" ok
  check c "numel" (numel [2, 3, 4] == 24 && numel [7] == 7 && numel [] == 1)
  -- matmul [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]]
  let A : Tensor.Matrix2D 2 2 Float := Tensor.ofFun2 fun i j => #[#[1.0, 2.0], #[3.0, 4.0]][i.val]![j.val]!
  let B : Tensor.Matrix2D 2 2 Float := Tensor.ofFun2 fun i j => #[#[5.0, 6.0], #[7.0, 8.0]][i.val]![j.val]!
  let C := Tensor.matmul A B
  check c "matmul entries" (approx (C.get2 0 0) 19 && approx (C.get2 0 1) 22 && approx (C.get2 1 0) 43 && approx (C.get2 1 1) 50)
  check c "get2/ofFun2 round trip" (approx (A.get2 1 0) 3 && approx (A.get2 0 1) 2)
  -- network forward: x=(2,1): fc1=[1,2], relu→[1,2], fc2 = 1+2+0.5 ... see below
  let x : Tensor.Vector 2 Float := fun i => if i.val == 0 then 2.0 else 1.0
  -- fc1: [2*1+1*(-1)+0, 2*0.5+1*2-1] = [1, 2]; relu: [1,2]; fc2: 1+2+0.5 = 3.5
  check c "forward value" (approx (netA.forward x 0) 3.5)
  let xn : Tensor.Vector 2 Float := fun i => if i.val == 0 then -3.0 else 1.0
  -- fc1: [-3-1, -1.5+2-1] = [-4, -0.5]; relu: [0,0]; fc2: 0.5
  check c "relu clips negatives" (approx (netA.forward xn 0) 0.5)
  -- concat theorem, checked numerically as well
  let h : LayerChain Float 2 2 := .consReLU {} (.nil 2)
  let t : LayerChain Float 2 1 := .consLinear ⟨fun _ _ => 1.0, fun _ => 0.0⟩ (.nil 1)
  check c "concat = composition" (approx ((h.concat t).run x 0) (t.run (h.run x) 0))
  -- native backend
  let m ← NativeTensor.create2D #[1, 2, 3, 4] 2 2
  let v ← NativeTensor.create1D #[2, 1]
  check c "native matmul 2x2·2" ((← (← NativeTensor.matmul m v).toArray) == #[4.0, 10.0])
  let m2 ← NativeTensor.create2D #[5, 6, 7, 8] 2 2
  check c "native matmul 2x2·2x2" ((← (← NativeTensor.matmul m m2).toArray) == #[19.0, 22.0, 43.0, 50.0])
  check c "native agrees with Lean matmul" ((← (← NativeTensor.matmul m m2).toArray)[1]! == C.get2 0 1)
  check c "native add" ((← (← NativeTensor.add m m2).toArray) == #[6.0, 8.0, 10.0, 12.0])
  check c "native sub" ((← (← NativeTensor.sub m2 m).toArray) == #[4.0, 4.0, 4.0, 4.0])
  let neg ← NativeTensor.create1D #[-1, 0, 2]
  check c "native relu" ((← (← neg.relu).toArray) == #[0.0, 0.0, 2.0])
  check c "native tanh(0)=0" ((← (← (← NativeTensor.create1D #[0]).tanh).toArray) == #[0.0])
  check c "native rank/dim" ((← m.rank) == 2 && (← m.dim 1) == 2 && (← v.rank) == 1)
  check c "shape mismatch → IO error" (← raises (NativeTensor.add m v))
  check c "matmul inner-dim mismatch → IO error" (← raises (NativeTensor.matmul m neg))
  check c "bad create2D size → IO error" (← raises (NativeTensor.create2D #[1, 2, 3] 2 2))
  check c "dim out of range → IO error" (← raises (m.dim 5))
  check c "CUDA rejected by native-c" (← raises (NativeTensor.create1D #[1] (.CUDA 0)))
  -- memory: allocate/drop many tensors (finalizer must not crash or leak unboundedly)
  for _ in [0:20000] do
    let t ← NativeTensor.create1D (Array.mkArray 64 1.0)
    let _ ← t.relu
  check c "20k alloc/free cycles" true
  let s ← c.get
  IO.println s!"tests passed: {s.pass}, failed: {s.fail}"
  return if s.fail == 0 then 0 else 1
