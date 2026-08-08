import LeanTorch

open LeanTorch.Core
open LeanTorch.Cert
open LeanTorch.FFI

/--
Main entrypoint executable for testing runtime initialization,
native LibTorch FFI connectivity, and formal verification engine sanity.
-/
def main : IO Unit := do
  IO.println "============================================================"
  IO.println "  LeanTorch-Cert: Verified Deep Learning Runtime (Lean 4)   "
  IO.println "============================================================"

  -- 1. Test Real Interval Arithmetic Engine
  let i1 : RealInterval := ⟨-1.5, 2.0, by linarith⟩
  let i2 : RealInterval := ⟨0.5, 3.0, by linarith⟩
  let i_sum := i1 + i2
  IO.println s!"[Interval Engine] [-1.5, 2.0] + [0.5, 3.0] = [{i_sum.lo}, {i_sum.hi}]"

  -- 2. Test Dependently-Typed Tensor Structure
  let v : Tensor.Vector 2 ℝ := fun i => match i.val with
    | 0 => 0.5
    | _ => -1.0
  let norm_sq := Tensor.dot v v
  IO.println s!"[Core Engine] Vector [0.5, -1.0] dot product with itself = {norm_sq}"

  -- 3. Verify Native LibTorch FFI Bridge (if library linked)
  try
    let native_arr ← NativeTensor.create1D #[0.5, -1.0] TorchDevice.CPU
    let arr_out ← native_arr.toArray
    native_arr.free
    IO.println s!"[FFI LibTorch] Successfully executed C++ roundtrip: {arr_out}"
  catch e =>
    IO.println s!"[FFI LibTorch] Native runtime skipped or unlinked: {e}"

  IO.println "✓ System initialization successful."
