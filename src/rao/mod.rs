//! Remedial action optimization.
//!
//! See `docs/src/rao/` for the mathematics.
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

pub mod automaton;
pub mod castor;
pub mod costly;
pub mod crac;
pub mod crac_json;
pub mod evaluate;
pub mod limits;
pub mod linear;
pub mod mnec;
pub mod parameters;
pub mod search;
pub mod usage;
pub mod validate;

pub use automaton::{AutomatonResult, simulate};
pub use castor::{PerimeterPlan, Plan, ScenarioPlan, StepsExecuted, run};
pub use costly::{Costly, CostlyOptions};
pub use crac::{
    Contingency, Crac, DOCUMENT_TYPE, DOCUMENT_VERSION, ElementaryAction, FlowCnec, Instant,
    InstantKind, RaUsageLimits, Range, RangeAction, RangeActionKind, RangeKind, RaoDocument, Side,
    State, Threshold, Unit, UsageRule,
};
pub use crac_json::{CracError, CracReport};
pub use evaluate::{
    AcOptions, CnecResult, FlowModel, Held, Network, PerimeterResult, Resolution, SecurityResult,
    ac_options, evaluate, evaluate_ac, evaluate_model, evaluate_split, evaluate_with,
};
pub use limits::{Budget, Limits};
pub use linear::{
    LinearOptions, LinearResult, LinearStatus, NetworkMut, ObjectiveUnit, Setpoint, TapModel,
    injection_setpoint, optimize,
};
pub use mnec::{Baseline, Mnec, MnecOptions};
pub use search::{
    SearchOptions, SearchResult, SecondPreventive, SecondPreventiveCondition, search,
    search_with_open,
};
pub use usage::Constrained;
pub use validate::{Validation, ValidationOptions, Verdict, validate};
