import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.Field.Basic

namespace LeanTorch.Cert

/--
A closed, bounded interval on the real numbers [lo, hi] ⊂ ℝ.
Guarantees via structural proof that lo ≤ hi.
-/
structure RealInterval where
  lo : ℝ
  hi : ℝ
  le : lo ≤ hi

namespace RealInterval

/-- Construct an interval with automatic proof obligation if lo ≤ hi holds. -/
def mk' (lo hi : ℝ) (h : lo ≤ hi) : RealInterval :=
  ⟨lo, hi, h⟩

/-- Membership predicate: x ∈ [lo, hi]. -/
def contains (I : RealInterval) (x : ℝ) : Prop :=
  I.lo ≤ x ∧ x ≤ I.hi

/-- Degenerate point interval [x, x]. -/
def point (x : ℝ) : RealInterval :=
  ⟨x, x, le_refl x⟩

/-- Width (diameter) of an interval. -/
def width (I : RealInterval) : ℝ :=
  I.hi - I.lo

theorem width_nonneg (I : RealInterval) : 0 ≤ I.width := by
  dsimp [width]
  linarith [I.le]

/-- Interval Addition: [a, b] + [c, d] = [a + c, b + d]. -/
def add (I1 I2 : RealInterval) : RealInterval :=
  ⟨I1.lo + I2.lo, I1.hi + I2.hi, add_le_add I1.le I2.le⟩

instance : Add RealInterval := ⟨add⟩

theorem contains_add {I1 I2 : RealInterval} {x y : ℝ}
    (hx : I1.contains x) (hy : I2.contains y) : (I1 + I2).contains (x + y) := by
  dsimp [contains, add] at *
  constructor
  · linarith [hx.1, hy.1]
  · linarith [hx.2, hy.2]

/-- Interval Negation: -[a, b] = [-b, -a]. -/
def neg (I : RealInterval) : RealInterval :=
  ⟨-I.hi, -I.lo, neg_le_neg I.le⟩

instance : Neg RealInterval := ⟨neg⟩

theorem contains_neg {I : RealInterval} {x : ℝ}
    (hx : I.contains x) : (-I).contains (-x) := by
  dsimp [contains, neg] at *
  constructor
  · linarith [hx.2]
  · linarith [hx.1]

/-- Interval Subtraction: I1 - I2 = I1 + (-I2). -/
def sub (I1 I2 : RealInterval) : RealInterval :=
  I1 + (-I2)

instance : Sub RealInterval := ⟨sub⟩

theorem contains_sub {I1 I2 : RealInterval} {x y : ℝ}
    (hx : I1.contains x) (hy : I2.contains y) : (I1 - I2).contains (x - y) := by
  have hy_neg := contains_neg hy
  have h_add := contains_add hx hy_neg
  exact h_add

/-- Scalar Multiplication: c * [a, b]. -/
def scale (c : ℝ) (I : RealInterval) : RealInterval :=
  if hc : 0 ≤ c then
    ⟨c * I.lo, c * I.hi, mul_le_mul_of_nonneg_left I.le hc⟩
  else
    have hc' : c ≤ 0 := le_of_not_ge hc
    ⟨c * I.hi, c * I.lo, by
      nlinarith [I.le, hc']⟩

theorem contains_scale {I : RealInterval} {c x : ℝ}
    (hx : I.contains x) : (scale c I).contains (c * x) := by
  dsimp [scale]
  split_ifs with hc
  · dsimp [contains] at *
    constructor
    · nlinarith [hx.1, hc]
    · nlinarith [hx.2, hc]
  · dsimp [contains] at *
    have hc' : c ≤ 0 := le_of_not_ge hc
    constructor
    · nlinarith [hx.2, hc']
    · nlinarith [hx.1, hc']

/-- Pointwise ReLU on RealInterval: [max(0, lo), max(0, hi)]. -/
def relu (I : RealInterval) : RealInterval :=
  ⟨max 0 I.lo, max 0 I.hi, by
    have h := I.le
    exact max_le_max (le_refl 0) h⟩

theorem contains_relu {I : RealInterval} {x : ℝ}
    (hx : I.contains x) : I.relu.contains (max 0 x) := by
  dsimp [contains, relu] at *
  constructor
  · exact max_le_max (le_refl 0) hx.1
  · exact max_le_max (le_refl 0) hx.2

/-- Pointwise LeakyReLU on RealInterval. -/
def leakyRelu (slope : ℝ) (I : RealInterval) : RealInterval :=
  let I_neg := scale slope ⟨I.lo, min 0 I.hi, by
    have h := I.le
    exact le_trans h (le_max_right 0 I.hi)⟩
  let I_pos := ⟨max 0 I.lo, max 0 I.hi, max_le_max (le_refl 0) I.le⟩
  I_neg + I_pos

end RealInterval

/--
An n-dimensional hyper-rectangle (interval vector) represented as a mapping `Fin n → RealInterval`.
-/
def IntervalVector (n : ℕ) :=
  Fin n → RealInterval

namespace IntervalVector

/-- Membership predicate for multidimensional vectors: v ∈ Box. -/
def contains {n : ℕ} (box : IntervalVector n) (v : Fin n → ℝ) : Prop :=
  ∀ i : Fin n, (box i).contains (v i)

/-- Pointwise addition of interval vectors. -/
def add {n : ℕ} (b1 b2 : IntervalVector n) : IntervalVector n :=
  fun i => b1 i + b2 i

instance {n : ℕ} : Add (IntervalVector n) := ⟨add⟩

theorem contains_add {n : ℕ} {b1 b2 : IntervalVector n} {u v : Fin n → ℝ}
    (hu : b1.contains u) (hv : b2.contains v) : (b1 + b2).contains (u + v) := by
  intro i
  exact RealInterval.contains_add (hu i) (hv i)

end IntervalVector

end LeanTorch.Cert
