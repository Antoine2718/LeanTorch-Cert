import Lake
open Lake DSL

package «LeanTorch» where
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩,
    ⟨`autoImplicit, false⟩
  ]

lean_lib «LeanTorch» where
  srcDir := "."

@[default_target]
lean_exe «leantorch» where
  root := `Main

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.8.0"
