import LeanTorch

open LeanTorch.Core
open LeanTorch.Cert

namespace Tests.Integration

/-- Integration Test 1: Tensor Matrix Multiplication Consistency -/
def testMatMul : Bool :=
  let A : Tensor.Matrix2D 2 2 ℝ := fun idx => match idx.val with
    | 0 => 1.0 | 1 => 2.0
    | 2 => 3.0 | _ => 4.0
  let B : Tensor.Matrix2D 2 2 ℝ := fun idx => match idx.val with
    | 0 => 1.0 | 1 => 0.0
    | 2 => 0.0 | _ => 1.0
  let C := Tensor.matmul A B
  (C ⟨0, by decide⟩ == 1.0) && (C ⟨3, by decide⟩ == 4.0)

/-- Integration Test 2: Interval Addition Containment -/
def testIntervalAdd : Bool :=
  let I1 : RealInterval := ⟨1.0, 3.0, by linarith⟩
  let I2 : RealInterval := ⟨-2.0, 0.5, by linarith⟩
  let I_sum := I1 + I2
  (I_sum.lo == -1.0) && (I_sum.hi == 3.5) && I_sum.contains 0.0

#eval testMatMul
#eval testIntervalAdd

end Tests.Integration
