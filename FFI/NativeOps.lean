import Lean
import LeanTorch.FFI.CTypes

namespace LeanTorch.FFI

open Lean

/--
Low-level Foreign Function Interface (FFI) declarations targeting `libtorch_bridge.so`.
These functions map directly to C-linkage symbols exposed by the LibTorch C++ wrapper.
-/

@[extern "leantorch_tensor_create_1d"]
opaque c_tensor_create_1d (data : @& Array Float) (size : Nat) (dtype : Int32) (device : Int32) : IO TorchTensorHandle

@[extern "leantorch_tensor_create_2d"]
opaque c_tensor_create_2d (data : @& Array Float) (rows : Nat) (cols : Nat) (dtype : Int32) (device : Int32) : IO TorchTensorHandle

@[extern "leantorch_tensor_free"]
opaque c_tensor_free (handle : TorchTensorHandle) : IO Unit

@[extern "leantorch_tensor_add"]
opaque c_tensor_add (a : TorchTensorHandle) (b : TorchTensorHandle) : IO TorchTensorHandle

@[extern "leantorch_tensor_sub"]
opaque c_tensor_sub (a : TorchTensorHandle) (b : TorchTensorHandle) : IO TorchTensorHandle

@[extern "leantorch_tensor_matmul"]
opaque c_tensor_matmul (a : TorchTensorHandle) (b : TorchTensorHandle) : IO TorchTensorHandle

@[extern "leantorch_tensor_relu"]
opaque c_tensor_relu (handle : TorchTensorHandle) : IO TorchTensorHandle

@[extern "leantorch_tensor_tanh"]
opaque c_tensor_tanh (handle : TorchTensorHandle) : IO TorchTensorHandle

@[extern "leantorch_tensor_to_array"]
opaque c_tensor_to_array (handle : TorchTensorHandle) : IO (Array Float)

@[extern "leantorch_tensor_get_rank"]
opaque c_tensor_get_rank (handle : TorchTensorHandle) : IO Nat

@[extern "leantorch_tensor_get_dim"]
opaque c_tensor_get_dim (handle : TorchTensorHandle) (dim_idx : Nat) : IO Nat

/--
Safe Lean 4 High-Level Managed Wrapper around native LibTorch C++ tensors.
Uses Lean's `IO` monad for memory-safe allocated handles and side effects.
-/
structure NativeTensor where
  handle : TorchTensorHandle

namespace NativeTensor

/-- Allocates a 1D double-precision native C++ float tensor. -/
def create1D (data : Array Float) (device : TorchDevice := TorchDevice.CPU) : IO NativeTensor := do
  let handle ← c_tensor_create_1d data data.size (TorchDtype.Float64.toCInt) device.toCInt
  pure { handle := handle }

/-- Allocates a 2D double-precision native C++ matrix tensor. -/
def create2D (data : Array Float) (rows cols : Nat) (device : TorchDevice := TorchDevice.CPU) : IO NativeTensor := do
  if data.size != rows * cols then
    throw (IO.userError s!"[LeanTorch FFI] Array size ({data.size}) does not match dimensions ({rows}x{cols}).")
  let handle ← c_tensor_create_2d data rows cols (TorchDtype.Float64.toCInt) device.toCInt
  pure { handle := handle }

/-- Destroys a native C++ tensor and releases underlying LibTorch memory. -/
def free (t : NativeTensor) : IO Unit := do
  c_tensor_free t.handle

/-- Executes C++ LibTorch tensor addition: `A + B`. -/
def add (a b : NativeTensor) : IO NativeTensor := do
  let handle ← c_tensor_add a.handle b.handle
  pure { handle := handle }

/-- Executes C++ LibTorch tensor subtraction: `A - B`. -/
def sub (a b : NativeTensor) : IO NativeTensor := do
  let handle ← c_tensor_sub a.handle b.handle
  pure { handle := handle }

/-- Executes C++ LibTorch matrix multiplication: `A × B`. -/
def matmul (a b : NativeTensor) : IO NativeTensor := do
  let handle ← c_tensor_matmul a.handle b.handle
  pure { handle := handle }

/-- Applies pointwise C++ LibTorch ReLU activation. -/
def relu (t : NativeTensor) : IO NativeTensor := do
  let handle ← c_tensor_relu t.handle
  pure { handle := handle }

/-- Applies pointwise C++ LibTorch Tanh activation. -/
def tanh (t : NativeTensor) : IO NativeTensor := do
  let handle ← c_tensor_tanh t.handle
  pure { handle := handle }

/-- Copies native C++ tensor contents back into a Lean 4 Float Array. -/
def toArray (t : NativeTensor) : IO (Array Float) := do
  c_tensor_to_array t.handle

/-- Retrieves the tensor rank (number of dimensions). -/
def rank (t : NativeTensor) : IO Nat := do
  c_tensor_get_rank t.handle

/-- Retrieves the size along a specific dimension index. -/
def dim (t : NativeTensor) (dimIdx : Nat) : IO Nat := do
  c_tensor_get_dim t.handle dimIdx

end NativeTensor

end LeanTorch.FFI
