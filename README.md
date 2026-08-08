# LeanTorch-Cert

> **A Dependently-Typed Architecture for Compiled Tensor Safety and Certified Neural Verification**

`LeanTorch-Cert` is a formal verification and safe deep learning framework that bridges continuous neural network optimization in **PyTorch** with interactive theorem proving in **Lean 4**.

---

## Key Features

1. **Compile-Time Tensor Shape Safety**:
   - Shape-indexed tensors `Tensor (shape : List ℕ) ℝ` powered by Lean 4 dependent types.
   - Elimination of runtime dimension mismatches (`RuntimeError: shape mismatch`) at compile-time.
   - Declarative DSL (`torch_net % [...]`) with automatic shape chaining.

2. **Exact Parameter Serialization**:
   - Float-to-Rational conversion ($\mathbb{F} \to \mathbb{Q} \to \mathbb{R}$) preserving mathematical precision.
   - Automated code generation emitting native Lean 4 formal structures.

3. **Formal Verification & Certification Engine**:
   - **Real Interval Arithmetic Engine**: Rigorous bounds propagation $[a, b] \subset \mathbb{R}$.
   - **Neural Lyapunov Certification**: Automated proofs of asymptotic stability for continuous non-linear systems $\dot{x} = f(x)$.
   - **Adversarial Robustness**: $L_\infty$ robustness verification via Interval Bound Propagation (IBP).

4. **CEGIS Training Pipeline**:
   - PyTorch Counterexample-Guided Inductive Synthesis (CEGIS) trainer for candidate Lyapunov functions $V_\theta(x)$.

---

## Directory Layout

```text
LeanTorch-Cert/
├── lakefile.lean               # Lean 4 package manifest & dependencies
├── LeanTorch.lean              # Master Lean library root
├── Main.lean                   # Runtime entrypoint
│
├── LeanTorch/                  # Core Lean 4 Formal Framework
│   ├── Core/                   # Tensors, Layers, Networks & Mathlib integration
│   ├── Elab/                   # Metaprogramming, DSL & Compile-time JSON Importer
│   ├── FFI/                    # LibTorch C-ABI bindings & Native C++ bridge
│   └── Cert/                   # Intervals, Semantics, Lyapunov & Robustness Proofs
│
├── python/                     # PyTorch Exporter & PINN Trainer
│   ├── leantorch_exporter/     # IR Schemas, FX Exporter, CEGIS Trainer
│   └── tests/                  # PyTest verification suite
│
├── examples/                   # Executable Lean 4 Certification Examples
│   ├── 01_CompileTimeShapes.lean
│   ├── 02_LyapunovCertification.lean
│   └── 03_RobustnessVerification.lean
│
└── tests/                      # Lean 4
```
## Quick start
### 1. Build the Lean 4 Framework
```bash
# Fetch dependencies (Mathlib 4) and build library
lake build
```
### 2. Run Python Export & CEGIS Loop
```bash
cd python
pip install -e .
pytest
```
