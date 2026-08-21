//! Remedial action optimization.
//!
//! See `plans/RAO_PLAN.md` for what this is and how it is meant to be built,
//! and `docs/src/rao/` for the mathematics.
//!
//! The layers, outermost first: [`castor`] decides which perimeters there are
//! and in what order, [`mod@search`] chooses the network actions of one
//! perimeter, and [`linear`] chooses the continuous set-points of one
//! candidate. [`mod@evaluate`] is what all three measure with, [`usage`]
//! decides which actions are on the table at all, [`limits`] caps how many may
//! be used, [`mnec`] is the one
//! constraint that is neither maximized nor hard, and [`mod@validate`]
//! re-measures the answer under a full AC power flow. [`crac`] and
//! [`crac_json`] are the data layer underneath.

pub mod crac;
pub mod crac_json;
pub mod evaluate;
pub mod linear;
pub mod automaton;
pub mod castor;
pub mod search;
pub mod mnec;
pub mod limits;
pub mod usage;
pub mod validate;

pub use crac::{
    RaoDocument, DOCUMENT_TYPE, DOCUMENT_VERSION,
    Contingency, Crac, ElementaryAction, FlowCnec, Instant, InstantKind, RaUsageLimits, Range,
    RangeAction, RangeActionKind, RangeKind, Side, State, Threshold, Unit, UsageRule,
};
pub use crac_json::{CracError, CracReport};
pub use automaton::{simulate, AutomatonResult};
pub use castor::{run, PerimeterPlan, Plan, ScenarioPlan};
pub use search::{search, SearchOptions, SearchResult};
pub use linear::{
    optimize, LinearOptions, LinearResult, LinearStatus, NetworkMut, ObjectiveUnit, Setpoint,
    TapModel,
};
pub use evaluate::{
    evaluate, evaluate_ac, evaluate_model, evaluate_with, AcOptions, CnecResult, FlowModel,
    Network, PerimeterResult, Resolution, SecurityResult,
};
pub use mnec::{Baseline, Mnec, MnecOptions};
pub use limits::{Budget, Limits};
pub use usage::Constrained;
pub use validate::{validate, Validation, ValidationOptions, Verdict};
