//! Bracketed root finding, shared by the event locators in
//! [`continuation`](crate::continuation) and gridoxide's `dynamics`.

/// Regula falsi with Illinois damping over `[lo, hi]`, given the bracketing
/// values `f_lo < 0 <= f_hi`.
///
/// Each evaluation costs a full corrector in continuation and a full Newton
/// solve in dynamics, so plain bisection would be wasteful; Illinois damping
/// keeps the superlinear rate without losing the bracket the way unmodified
/// regula falsi does when one endpoint stalls.
pub struct Illinois {
    pub lo: f64,
    pub hi: f64,
    pub f_lo: f64,
    pub f_hi: f64,
}

impl Illinois {
    /// The next trial point.
    pub fn next(&self) -> f64 {
        let denom = self.f_hi - self.f_lo;
        if !denom.is_finite() || denom == 0.0 {
            return 0.5 * (self.lo + self.hi);
        }
        let x = self.hi - self.f_hi * (self.hi - self.lo) / denom;
        // Never leave the bracket, and never stall on an endpoint.
        let (a, b) = (self.lo.min(self.hi), self.lo.max(self.hi));
        let margin = 1e-3 * (b - a);
        x.clamp(a + margin, b - margin)
    }

    /// Narrows the bracket with the value measured at `x`.
    pub fn update(&mut self, x: f64, f_x: f64) {
        if f_x >= 0.0 {
            self.hi = x;
            self.f_hi = f_x;
            self.f_lo *= 0.5; // Illinois: halve the stale endpoint's weight
        } else {
            self.lo = x;
            self.f_lo = f_x;
            self.f_hi *= 0.5;
        }
    }

    pub fn width(&self) -> f64 {
        (self.hi - self.lo).abs()
    }
}
