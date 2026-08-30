//! A dynamics case of any size, for the gates that need one.
//!
//! A ring of alternating generator and load buses: sparse the way a real
//! network is, and unlike a star it gives no bus a dense Jacobian column. Every
//! generator carries a sixth-order machine with an exciter and a governor, so
//! each unit contributes ten differential states on top of the network's two
//! per bus.
//!
//! Two knobs matter to the callers here:
//!
//! - **size**, because the whole point of the sparse eigensolver is the region
//!   where the dense one cannot start; and
//! - **`spread`**, which scales each machine's inertia by `1 + spread·i/n`. A
//!   ring of *identical* machines is circulant, and its electromechanical modes
//!   come in near-exact degenerate pairs — the one case a Krylov method is bad
//!   at, since a Krylov space holds one vector per invariant direction. Both
//!   cases are worth gating, so both are reachable: `spread = 0` builds the
//!   degenerate ring on purpose.
//!
//! Shared verbatim with `examples/smallsignal_scale.rs`, which includes this
//! file directly rather than keeping a second copy that could drift.

#![allow(dead_code)]

use gridoxide::dynamics::json::{
    AvrSpec, DynamicsData, DynamicsDocument, EventSpec, GovSpec, MachineSpec, UnitSpec,
};
use gridoxide::dynamics::models::avr::SexsParams;
use gridoxide::dynamics::models::gov::Tgov1Params;
use gridoxide::dynamics::models::machine::GenRoundParams;
use gridoxide::dynamics::models::Limits;
use gridoxide::json::NetworkData;
use gridoxide::types::{Bus, BusType, Line};

pub const S_BASE: f64 = 100.0;
/// Differential states per generating unit: machine 6, exciter 2, governor 2.
pub const STATES_PER_UNIT: usize = 10;

fn bus(idx: usize, bus_type: BusType, p: f64, q: f64) -> Bus {
    Bus {
        idx,
        bus_type,
        voltage_mag: 1.0,
        voltage_ang: 0.0,
        p_spec: p,
        q_spec: q,
        q_min: f64::NEG_INFINITY,
        q_max: f64::INFINITY,
        u_rated: 1.0,
        zip_terms: Vec::new(),
    }
}

/// A ring of `n_bus` buses, alternating generation and load.
///
/// `spread` breaks the ring's symmetry through the machine inertias; see the
/// module doc for why that is a parameter rather than a constant.
pub fn ring(n_bus: usize, spread: f64) -> DynamicsDocument {
    let mut buses = Vec::with_capacity(n_bus);
    let mut units = Vec::new();
    for i in 0..n_bus {
        let generating = i % 2 == 0;
        let kind = if i == 0 {
            BusType::Slack
        } else if generating {
            BusType::PV
        } else {
            BusType::PQ
        };
        // Modest injections and a short electrical distance between
        // neighbours: a long ring carrying heavy flow is a *power flow*
        // problem, and a base case that fails to solve would put a ceiling in
        // front of the thing being measured.
        let p = if generating { 0.30 } else { -0.30 };
        buses.push(bus(i, kind, p, if generating { 0.0 } else { -0.08 }));

        if generating {
            let h = 5.0 * (1.0 + spread * i as f64 / n_bus as f64);
            units.push(UnitSpec {
                id: format!("G{i}"),
                bus: i,
                p: None,
                q: None,
                machine: MachineSpec::GenRound(GenRoundParams {
                    h,
                    d: 1.0,
                    ra: 0.003,
                    xd: 1.8,
                    xq: 1.7,
                    xdp: 0.30,
                    xqp: 0.55,
                    xdpp: 0.22,
                    xqpp: 0.25,
                    xl: 0.15,
                    td0p: 8.0,
                    tq0p: 0.4,
                    td0pp: 0.03,
                    tq0pp: 0.05,
                    mbase: S_BASE,
                }),
                avr: Some(AvrSpec::Sexs(SexsParams {
                    k: 200.0,
                    ta: 0.1,
                    tb: 1.0,
                    te: 0.05,
                    limits: Limits::NONE,
                })),
                gov: Some(GovSpec::Tgov1(Tgov1Params {
                    r: 0.05,
                    t1: 0.5,
                    t2: 1.0,
                    t3: 5.0,
                    dt: 0.0,
                    limits: Limits::NONE,
                    rate: Limits::NONE,
                })),
                pss: None,
            });
        }
    }

    let lines = (0..n_bus)
        .map(|i| Line {
            from: i,
            to: (i + 1) % n_bus,
            r: 0.002,
            x: 0.02,
            b_shunt: 0.0,
            g_shunt: 0.0,
        })
        .collect();

    DynamicsDocument {
        network: NetworkData { buses, lines },
        dynamics: DynamicsData {
            s_base: S_BASE,
            f_nom: 50.0,
            speed_voltages: false,
            units,
            loads: Vec::new(),
            fixed_buses: Vec::new(),
            events: Vec::new(),
            relays: Vec::new(),
        },
    }
}

/// The same ring with a bolted fault applied at `t = 1 s` and cleared 80 ms
/// later — what the time-domain cross-check needs.
pub fn ring_with_fault(n_bus: usize, spread: f64) -> DynamicsDocument {
    let mut document = ring(n_bus, spread);
    document.dynamics.events = vec![
        EventSpec::BusFault { t: 1.0, bus: 1, y: None },
        EventSpec::ClearFault { t: 1.08, bus: 1 },
    ];
    document
}
