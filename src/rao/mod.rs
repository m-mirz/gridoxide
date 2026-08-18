//! Remedial action optimization.
//!
//! See `plans/RAO_PLAN.md` for what this is and how it is meant to be built.
//! At present this module carries the **data layer**: the CRAC domain model and
//! the readers that populate it. The optimizer itself is not here yet.

pub mod crac;
pub mod crac_json;

pub use crac::{
    RaoDocument, DOCUMENT_TYPE, DOCUMENT_VERSION,
    Contingency, Crac, ElementaryAction, FlowCnec, Instant, InstantKind, RaUsageLimits, Range,
    RangeAction, RangeActionKind, RangeKind, Side, State, Threshold, Unit, UsageRule,
};
pub use crac_json::{CracError, CracReport};
