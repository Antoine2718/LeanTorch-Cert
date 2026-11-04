namespace LeanTorch.FFI

/-- Execution device. The default `native-c` backend only supports `CPU`. -/
inductive TorchDevice where
  | CPU
  | CUDA (id : UInt32)
  deriving Inhabited, Repr, BEq

/-- Integer code passed over the C ABI (`0` = CPU, `100 + id` = CUDA `id`). -/
def TorchDevice.toCode : TorchDevice → UInt32
  | .CPU    => 0
  | .CUDA i => 100 + i

/-- Opaque native tensor (a Lean *external object*; freed by Lean's reference counting). -/
opaque NativeTensorPointed : NonemptyType
def NativeTensor : Type := NativeTensorPointed.type
instance : Nonempty NativeTensor := NativeTensorPointed.property

end LeanTorch.FFI
