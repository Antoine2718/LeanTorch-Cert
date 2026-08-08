import Lean

namespace LeanTorch.FFI

/--
Opaque handle pointing to a native LibTorch C++ `at::Tensor` instance in memory.
Memory management is controlled via reference counting on the C++ side and explicit
finalizer calls or IO encapsulation on the Lean 4 side.
-/
opaque TorchTensorPointed : PointedType
def TorchTensorHandle : Type := TorchTensorPointed.type

/--
Supported execution devices for LibTorch hardware acceleration.
-/
inductive TorchDevice where
  | CPU  : TorchDevice
  | CUDA : UInt32 → TorchDevice
  deriving Inhabited, Repr

/-- Converts a `TorchDevice` into its integer code for C FFI boundary passing. -/
def TorchDevice.toCInt : TorchDevice → Int32
  | TorchDevice.CPU => 0
  | TorchDevice.CUDA deviceId => 100 + deviceId.toInt32

/--
Supported numerical dtypes for underlying C++ tensor buffers.
-/
inductive TorchDtype where
  | Float32 : TorchDtype
  | Float64 : TorchDtype
  | Int32   : TorchDtype
  | Int64   : TorchDtype
  deriving Inhabited, Repr

/-- Converts a `TorchDtype` into its integer code for C FFI boundary passing. -/
def TorchDtype.toCInt : TorchDtype → Int32
  | TorchDtype.Float32 => 0
  | TorchDtype.Float64 => 1
  | TorchDtype.Int32   => 2
  | TorchDtype.Int64   => 3

end LeanTorch.FFI
