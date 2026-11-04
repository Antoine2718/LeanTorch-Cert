import Mathlib.Data.Real.Basic
import LeanTorch.Cert.RealInstances
namespace LeanTorch.Cert
noncomputable section

/-- Closed interval `[lo, hi] ⊂ ℝ`, with `lo ≤ hi` carried as a proof. -/
structure RealInterval where
  lo : ℝ
  hi : ℝ
  le : lo ≤ hi

namespace RealInterval

/-- Membership `x ∈ [lo, hi]` (an `abbrev`, so `.1`/`.2`/`⟨_,_⟩` work directly). -/
abbrev contains (I : RealInterval) (x : ℝ) : Prop := I.lo ≤ x ∧ x ≤ I.hi

def point (x : ℝ) : RealInterval := ⟨x, x, le_refl x⟩
instance : OfNat RealInterval 0 := ⟨point 0⟩

def add (I J : RealInterval) : RealInterval := ⟨I.lo + J.lo, I.hi + J.hi, add_le_add I.le J.le⟩
instance : Add RealInterval := ⟨add⟩

theorem contains_add {I J : RealInterval} {x y : ℝ}
    (hx : I.contains x) (hy : J.contains y) : (I + J).contains (x + y) :=
  ⟨add_le_add hx.1 hy.1, add_le_add hx.2 hy.2⟩

/-- `c · [lo, hi]`, endpoints swapped when `c < 0`. -/
def scale (c : ℝ) (I : RealInterval) : RealInterval :=
  if hc : 0 ≤ c then ⟨c * I.lo, c * I.hi, mul_le_mul_of_nonneg_left I.le hc⟩
  else ⟨c * I.hi, c * I.lo, mul_le_mul_of_nonpos_left I.le (le_of_lt (not_le.mp hc))⟩

theorem contains_scale {I : RealInterval} {c x : ℝ} (hx : I.contains x) :
    (scale c I).contains (c * x) := by
  unfold scale
  by_cases hc : 0 ≤ c
  · rw [dif_pos hc]
    exact ⟨mul_le_mul_of_nonneg_left hx.1 hc, mul_le_mul_of_nonneg_left hx.2 hc⟩
  · rw [dif_neg hc]
    have hc' : c ≤ 0 := le_of_lt (not_le.mp hc)
    exact ⟨mul_le_mul_of_nonpos_left hx.2 hc', mul_le_mul_of_nonpos_left hx.1 hc'⟩

/-- Pointwise ReLU (monotone, so endpoints map to endpoints). -/
def relu (I : RealInterval) : RealInterval :=
  ⟨max 0 I.lo, max 0 I.hi, max_le_max (le_refl 0) I.le⟩

theorem contains_relu {I : RealInterval} {x : ℝ} (hx : I.contains x) :
    I.relu.contains (max 0 x) :=
  ⟨max_le_max (le_refl 0) hx.1, max_le_max (le_refl 0) hx.2⟩

/-- LeakyReLU is monotone iff its slope is nonnegative. -/
theorem leaky_mono {s : ℝ} (hs : 0 ≤ s) {x y : ℝ} (h : x ≤ y) : leakyFn s x ≤ leakyFn s y := by
  unfold leakyFn
  by_cases hx : 0 ≤ x <;> by_cases hy : 0 ≤ y <;> simp only [hx, hy, ↓reduceIte]
  · exact h
  · exact absurd (le_trans hx h) hy
  · have hx' : x ≤ 0 := le_of_lt (not_le.mp hx)
    have h1 : s * x ≤ s * 0 := mul_le_mul_of_nonneg_left hx' hs
    rw [mul_zero] at h1
    exact le_trans h1 hy
  · exact mul_le_mul_of_nonneg_left h hs

def leaky (s : ℝ) (hs : 0 ≤ s) (I : RealInterval) : RealInterval :=
  ⟨leakyFn s I.lo, leakyFn s I.hi, leaky_mono hs I.le⟩

theorem contains_leaky {s : ℝ} (hs : 0 ≤ s) {I : RealInterval} {x : ℝ} (hx : I.contains x) :
    (leaky s hs I).contains (leakyFn s x) :=
  ⟨leaky_mono hs hx.1, leaky_mono hs hx.2⟩

end RealInterval

/-- Axis-aligned box in `ℝⁿ`. -/
def IntervalVector (n : Nat) : Type := Fin n → RealInterval

namespace IntervalVector
abbrev contains {n : Nat} (B : IntervalVector n) (v : Fin n → ℝ) : Prop := ∀ i, (B i).contains (v i)
end IntervalVector

end -- noncomputable section
end LeanTorch.Cert
