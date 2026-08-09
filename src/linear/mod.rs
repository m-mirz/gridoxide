//! Linearized power-flow methods: the two different things the literature and
//! the reference tools both call "linear power flow".
//!
//! - [`btheta`] is the **DC approximation**: real-valued, `B θ = P`, voltage
//!   magnitudes asserted to 1 p.u. and resistance discarded. It answers
//!   "how does active power divide between these paths", exactly, in one
//!   factorization and no iterations. This is what lightsim2grid,
//!   powsybl-open-loadflow and pandapower's `rundcpp` mean by DC power flow.
//! - [`impedance`] is the **constant-admittance linearization**: complex,
//!   `Y' U = I`, every load replaced by the admittance that would draw its
//!   rated power at `|V| = 1`. It keeps resistance and produces voltage
//!   *magnitudes*, which Bθ cannot. This is what power-grid-model means by
//!   `CalculationMethod.linear`.
//!
//! Neither approximates the other. Bθ is the transmission tool: on a meshed
//! network with small angle differences and `r << x` its flows are close to
//! the AC answer, and its exactness under linearity is what makes the
//! sensitivity factors in [`sensitivity`] meaningful. The constant-admittance
//! form is the distribution tool: it tracks the voltage drop along a feeder,
//! which is the quantity that actually matters there and the one Bθ throws
//! away by construction.
//!
//! **Not to be confused with [`crate::dc`]**, which despite the name is not an
//! approximation at all — it is the genuine HVDC network solver
//! (`DcBus`/`DcLine`/`solve_dc_network`) that `crate::cgmes` uses to resolve
//! converter stations. That module models real DC hardware; this one
//! approximates AC.

pub mod btheta;
pub mod impedance;
pub mod sensitivity;

pub use btheta::{
    dc_branches, dc_power_flow, DcBranch, DcIslandReport, DcIslandStatus, DcSolution,
};
pub use impedance::{linear_power_flow, LinearIslandReport, LinearIslandStatus, LinearReport};
pub use sensitivity::{DcSensitivity, DenseMatrix};

/// Which terms the DC susceptance keeps.
///
/// Both variants are powsybl-open-loadflow's `DcApproximationType`
/// (`dc/equations/AbstractClosedBranchDcFlowEquationTerm.computePower`); no
/// other reference tool offers the choice, and every one of them uses what is
/// spelled [`IgnoreR`](Self::IgnoreR) here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DcApproximation {
    /// `b = 1/x`. The textbook DC approximation, and what MATPOWER's
    /// `makeBdc`, pandapower's `rundcpp` and lightsim2grid all compute.
    /// Discards resistance entirely, which is the assumption the whole DC
    /// model rests on anyway.
    #[default]
    IgnoreR,
    /// `b = x/(r² + x²)`, i.e. `−Im(y_series)`: the true series susceptance,
    /// with resistance retained in the denominator where it belongs.
    ///
    /// Strictly the better linearization of the two — it is the exact
    /// coefficient of `sin δ` in the AC flow equation, whereas `1/x` is that
    /// coefficient only in the limit `r → 0`. The gap is negligible on
    /// transmission (`r/x ≈ 0.1` gives a 1% difference) and material on
    /// distribution feeders, where `r/x` can exceed 1. It is not the default
    /// only because every tool worth comparing against uses `1/x`, and a
    /// default that silently disagrees with all of them would make every
    /// cross-check look like a bug.
    IgnoreG,
}

/// Configuration for the DC (Bθ) solver and for [`sensitivity::DcSensitivity`].
///
/// Deliberately small: DC has no tolerance and no iteration limit, because it
/// is a direct solve. The only real choices are which susceptance to use and
/// whether transformer ratios participate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DcOptions {
    pub approximation: DcApproximation,
    /// Whether an off-nominal transformer ratio `k = |tap|` divides the
    /// branch susceptance (`b/k`). True by default, matching MATPOWER
    /// (`b ./ tap`) and powsybl (`useTransformerRatio`). Setting it false
    /// reproduces the cruder "ratios are all 1" approximation some
    /// contingency tools still assume.
    pub use_transformer_ratio: bool,
}

impl Default for DcOptions {
    fn default() -> Self {
        Self { approximation: DcApproximation::default(), use_transformer_ratio: true }
    }
}
