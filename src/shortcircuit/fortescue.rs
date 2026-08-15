//! Symmetrical components — the sequence view of a short-circuit result.
//!
//! The solve happens in the phase domain (see the module docs for why), but
//! the phase domain is not the vocabulary short-circuit analysis is normally
//! written in. IEC 60909 states its formulae in terms of \\(Z_0\\), \\(Z_1\\),
//! \\(Z_2\\), and powsybl's short-circuit API models its results as
//! `FortescueValue` — zero, positive and negative components, with the
//! three-phase values *derived* from them. This module supplies that view.
//!
//! The transform is the standard one, \\(A^{-1}\\) applied to the phase
//! vector:
//!
//! \\[
//!   \begin{pmatrix} X_0 \\\\ X_1 \\\\ X_2 \end{pmatrix}
//!   = \frac{1}{3}
//!   \begin{pmatrix}
//!     1 & 1 & 1 \\\\
//!     1 & a & a^2 \\\\
//!     1 & a^2 & a
//!   \end{pmatrix}
//!   \begin{pmatrix} X_a \\\\ X_b \\\\ X_c \end{pmatrix},
//!   \qquad a = e^{j 2\pi/3}
//! \\]
//!
//! It is the exact inverse of the `A` that
//! [`network::fortescue_to_phase`](crate::network) builds the phase-domain
//! Y-bus with, so a result projected here and pushed back through that one
//! returns where it started.
//!
//! # Reading the output
//!
//! The sequence view is where a fault type's signature is legible at a glance,
//! which is most of why it is worth having:
//!
//! - A **three-phase** fault is balanced, so only the positive sequence is
//!   non-negligible.
//! - A **single-phase-to-ground** fault excites all three roughly equally —
//!   its current path runs through the zero-sequence network, which is why
//!   transformer winding configuration matters so much to it.
//! - A **two-phase** fault (clear of ground) has **no zero-sequence
//!   component**: with no ground return there is nowhere for zero-sequence
//!   current to go. That is a useful check on a result.
//! - A **two-phase-to-ground** fault excites all three.

use num_complex::Complex;

/// One quantity resolved into symmetrical components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequenceValue {
    /// Zero sequence — equal in all three phases, and the component that
    /// needs a ground return path to flow at all.
    pub zero: Complex<f64>,
    /// Positive sequence — the balanced, normal-rotation component.
    pub positive: Complex<f64>,
    /// Negative sequence — balanced but reverse-rotating; nonzero only for an
    /// unbalanced condition.
    pub negative: Complex<f64>,
}

impl SequenceValue {
    /// Projects a phase triple `[a, b, c]` onto its symmetrical components.
    pub fn from_phase(abc: &[Complex<f64>; 3]) -> Self {
        let a = Complex::from_polar(1.0, 2.0 * std::f64::consts::PI / 3.0);
        let a2 = a * a;
        let third = 1.0 / 3.0;
        Self {
            zero: (abc[0] + abc[1] + abc[2]) * third,
            positive: (abc[0] + a * abc[1] + a2 * abc[2]) * third,
            negative: (abc[0] + a2 * abc[1] + a * abc[2]) * third,
        }
    }

    /// Rebuilds the phase triple. The exact inverse of
    /// [`from_phase`](Self::from_phase).
    pub fn to_phase(self) -> [Complex<f64>; 3] {
        let a = Complex::from_polar(1.0, 2.0 * std::f64::consts::PI / 3.0);
        let a2 = a * a;
        [
            self.zero + self.positive + self.negative,
            self.zero + a2 * self.positive + a * self.negative,
            self.zero + a * self.positive + a2 * self.negative,
        ]
    }
}

/// Projects every node's voltage onto symmetrical components, in node order.
pub fn node_sequences(u_bus: &[[Complex<f64>; 3]]) -> Vec<SequenceValue> {
    u_bus.iter().map(SequenceValue::from_phase).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn balanced(magnitude: f64) -> [Complex<f64>; 3] {
        [0, 1, 2].map(|p| Complex::from_polar(magnitude, super::super::PHASE_ANG[p]))
    }

    /// A balanced set is pure positive sequence — the defining property, and
    /// the one that pins the rotation direction. Getting `a` and `a²` the
    /// wrong way round here would swap positive and negative, which every
    /// other test would still pass.
    #[test]
    fn a_balanced_set_is_pure_positive_sequence() {
        let s = SequenceValue::from_phase(&balanced(1.0));
        assert!(s.positive.norm() > 0.999, "positive = {}", s.positive);
        assert!(s.zero.norm() < 1e-12, "zero = {}", s.zero);
        assert!(s.negative.norm() < 1e-12, "negative = {}", s.negative);
    }

    /// Three equal phasors with no rotation between them are pure zero
    /// sequence.
    #[test]
    fn an_equal_set_is_pure_zero_sequence() {
        let x = Complex::new(0.4, -0.2);
        let s = SequenceValue::from_phase(&[x, x, x]);
        assert!((s.zero - x).norm() < 1e-15);
        assert!(s.positive.norm() < 1e-15);
        assert!(s.negative.norm() < 1e-15);
    }

    #[test]
    fn the_projection_round_trips() {
        let abc = [
            Complex::new(0.9, 0.1),
            Complex::new(-0.3, -0.8),
            Complex::new(-0.5, 0.6),
        ];
        let back = SequenceValue::from_phase(&abc).to_phase();
        for p in 0..3 {
            assert!((abc[p] - back[p]).norm() < 1e-14, "phase {p}: {} vs {}", abc[p], back[p]);
        }
    }
}
