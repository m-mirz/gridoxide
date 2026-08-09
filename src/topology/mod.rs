//! Network topology: the node/switch graph, and the reductions over it.
//!
//! Two layers, with a deliberate separation between them.
//!
//! [`reduction`] is the older one and is unchanged: union-find, group
//! compaction, and the zero-impedance clamp. It answers "these things are
//! electrically the same point, collapse them", and it is importer-agnostic
//! because both importers can reach a network needing it.
//!
//! [`model`] and [`bus_view`] are the newer one. They hold a **node-breaker**
//! description — connectivity nodes and the switches between them — and derive
//! an electrical bus view from it under a [`RetentionPolicy`]. That is the
//! layer that lets a switch keep its identity instead of being consumed at
//! import, which is what `plans/NODE_BREAKER_PLAN.md` is about.
//!
//! The distinction that matters: `reduction` merges *unconditionally*, so a
//! merged connection is gone. `bus_view` merges *selectively*, so a retained
//! switch survives into the solver as an element with a flow and a state. The
//! default policy is [`RetentionPolicy::MergeAll`], which reproduces the older
//! behavior exactly — and is required to, since every CGMES fixture asserts
//! against it.
//!
//! # Index spaces
//!
//! [`NodeIdx`] and [`BusIdx`] are different things and this module keeps them
//! different in the type system. `src/cgmes.rs` carries a comment describing a
//! real bug where pre-merge and post-merge indices were confused and
//! `AngleRefTopologicalNode` resolution picked an unrelated node; both were
//! `usize`, so nothing caught it. Making `bus_of[node]` the only well-typed
//! direction removes that whole class of mistake.

pub mod bus_view;
pub mod model;
pub mod reduction;

pub use bus_view::{bus_view, BusView, RetainedSwitch, RetentionPolicy};
pub use model::{NodeBreakerTopology, NodeIdx, Switch, SwitchIdx, SwitchKind};

// The reduction layer's items keep their historical `topology::` paths, so no
// existing caller moves. `plans/NODE_BREAKER_PLAN.md` §3 asks for exactly this
// ("keeping `UnionFind`/`merge_groups`/`clamp_branch_impedance` where they are
// as `topology::reduction`").
pub use reduction::{
    clamp_branch_impedance, merge_groups, union_all, UnionFind, IDEAL_CONNECTION_Y,
    ZERO_IMPEDANCE_THRESHOLD,
};
pub(crate) use reduction::ideal_connection_z;
