import Lake
open Lake DSL System

package «LeanTorch» where
  leanOptions := #[⟨`autoImplicit, false⟩]

-- BEGIN MATHLIB
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.15.0"
-- END MATHLIB

/-! Native tensor backend (pure C, no dependencies) -/

target leantorch_native.o pkg : FilePath := do
  let oFile := pkg.buildDir / "ffi" / "leantorch_native.o"
  let srcJob ← inputTextFile <| pkg.dir / "ffi" / "leantorch_native.c"
  let weakArgs := #["-I", (← getLeanIncludeDir).toString]
  buildO oFile srcJob weakArgs #["-fPIC", "-O2"] "cc" getLeanTrace

extern_lib libleantorch_native pkg := do
  let o ← leantorch_native.o.fetch
  buildStaticLib (pkg.nativeLibDir / nameToStaticLib "leantorch_native") #[o]

/-! Mathlib-free libraries: tensors, networks, DSL, importer, FFI -/

@[default_target]
lean_lib LeanTorchCore where
  roots := #[`LeanTorch.Core, `LeanTorch.Elab, `LeanTorch.FFI]

@[default_target]
lean_lib Examples where
  roots := #[`Examples.CompileTimeShapes, `Examples.ImportModel]

@[default_target]
lean_exe leantorch where
  root := `Main

@[test_driver]
lean_exe leantorch_tests where
  root := `Tests.Main

-- BEGIN MATHLIB
/-! ## Certification libraries (need Mathlib) -/

@[default_target]
lean_lib LeanTorch where
  roots := #[`LeanTorch]

@[default_target]
lean_lib CertExamples where
  roots := #[`Examples.Lyapunov, `Examples.Robustness]
-- END MATHLIB
