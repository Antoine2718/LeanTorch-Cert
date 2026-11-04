import LeanTorch.FFI.CTypes
/-!
Bindings to the native tensor backend (`ffi/leantorch_native.c`, or the optional
libtorch bridge in `ffi/torch/`). All operations are in `IO`; shape errors are
reported as `IO.userError`, never as crashes.
-/
namespace LeanTorch.FFI

@[extern "leantorch_backend_name"] opaque backendName (u : Unit) : String

@[extern "leantorch_tensor_create_1d"]
opaque NativeTensor.create1DRaw (data : @& Array Float) (device : UInt32) : IO NativeTensor
@[extern "leantorch_tensor_create_2d"]
opaque NativeTensor.create2DRaw (data : @& Array Float) (rows cols : @& Nat) (device : UInt32) : IO NativeTensor
@[extern "leantorch_tensor_add"]    opaque NativeTensor.add    (a b : @& NativeTensor) : IO NativeTensor
@[extern "leantorch_tensor_sub"]    opaque NativeTensor.sub    (a b : @& NativeTensor) : IO NativeTensor
@[extern "leantorch_tensor_matmul"] opaque NativeTensor.matmul (a b : @& NativeTensor) : IO NativeTensor
@[extern "leantorch_tensor_relu"]   opaque NativeTensor.relu   (a : @& NativeTensor) : IO NativeTensor
@[extern "leantorch_tensor_tanh"]   opaque NativeTensor.tanh   (a : @& NativeTensor) : IO NativeTensor
@[extern "leantorch_tensor_to_array"] opaque NativeTensor.toArray (a : @& NativeTensor) : IO (Array Float)
@[extern "leantorch_tensor_rank"]   opaque NativeTensor.rank   (a : @& NativeTensor) : IO Nat
@[extern "leantorch_tensor_dim"]    opaque NativeTensor.dim    (a : @& NativeTensor) (i : @& Nat) : IO Nat

namespace NativeTensor

/-- 1-D float64 tensor. -/
def create1D (data : Array Float) (device : TorchDevice := .CPU) : IO NativeTensor :=
  create1DRaw data device.toCode

/-- `rows × cols` float64 matrix (row-major data). -/
def create2D (data : Array Float) (rows cols : Nat) (device : TorchDevice := .CPU) : IO NativeTensor :=
  create2DRaw data rows cols device.toCode

end NativeTensor
end LeanTorch.FFI
