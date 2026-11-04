# LeanTorch-Cert

Shape-checked neural networks in Lean 4, an exact PyTorch → Lean importer, and **machine-checked
interval-bound-propagation (IBP) certificates** over ℝ.

## Quick start
```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
lake update # first time only: resolves Mathlib and writes lake-manifest.json
lake exe cache get # prebuilt Mathlib (otherwise Mathlib compiles from source: very slow)
lake build # builds everything, incl. the proofs in LeanTorch/Cert and Examples
lake exe leantorch # demo: Lean forward pass + native C tensor backend
lake test     
cd python && pip install -e ".[test]" && pytest
```
Toolchain: Lean 4.15.0 / Mathlib `v4.15.0` (pinned in `lean-toolchain` and `lakefile.lean`).

## Layout
| Path | Needs Mathlib | Content |
|---|---|---|
| `LeanTorch/Core` | no | `Tensor`, `DenseLayer`, `LayerChain` (adjacent widths are checked by the type system), generic over the scalar type: executable over `Float`, provable over `ℝ`. Theorem `run_concat`. |
| `LeanTorch/Elab` | no | `torch_net % [...]` DSL, `#assert_network_shape`, `#import_torch_model` (reads the exporter's JSON at elaboration time, exact rational weights). |
| `LeanTorch/FFI`, `ffi/` | no | Pure-C CPU tensor backend as Lean external objects (no manual free; shape errors are `IO` errors). libtorch bridge in `ffi/torch/` is a legacy prototype, **not functional** (see its README). |
| `LeanTorch/Cert` | yes | `RealInterval`, `evalChainIBP`, theorem **`evalChainIBP_sound`**, `verify_robustness_via_ibp`, `LyapunovConditions`. |
| `python/` | – | `leantorch_exporter`: FX exporter (exact rationals), exact `Fraction` IBP (`ibp.py`), CEGIS Lyapunov trainer. |

## Workflow
1. `ModelExporter(model, x).build_spec("m")` → `spec.write_json("m.json")` (exact dyadic rationals; no rounding).
2. Lean: `#import_torch_model_noncomputable "m.json" as MNet : ℝ`.
3. Certify: `python/leantorch_exporter/ibp.py` gives exact bounds; `verify_robustness_via_ibp` turns a returned box into a theorem.

## What is and is not guaranteed
* Certificates are about the **real-number** network defined by the exported weights, not about float32/float64 PyTorch inference (rounding is not modelled).
* IBP supports Linear, ReLU, LeakyReLU (slope ≥ 0). `tanh`/Sigmoid return `none` (never an unsound box).
* `LyapunovConditions` records the pointwise Lyapunov inequalities. Lean does **not** yet check that the supplied `gradV` is `∇V`, and the structure alone does not prove stability.
* The Python Lyapunov trainer is empirical (no counterexample found ≠ proof).
* The libtorch bridge and CUDA are not available (the native backend is CPU float64, rank 1–2).
