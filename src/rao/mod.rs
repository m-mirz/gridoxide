//! Remedial action optimization.
//!
//! See `plans/RAO_PLAN.md` for what this is and how it is meant to be built.
//! At present this module carries the **data layer**: the CRAC domain model and
//! the readers that populate it. The optimizer itself is not here yet.

pub mod crac;
pub mod crac_json;
pub mod evaluate;
pub mod linear;
pub mod automaton;
pub mod castor;
pub mod search;
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
pub use validate::{validate, Validation, ValidationOptions, Verdict};
