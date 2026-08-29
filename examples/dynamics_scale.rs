//! How an RMS run scales.
//!
//! Gate G10 of `plans/RMS_PLAN.md`: a large case runs to completion, and the
//! cost measures that matter are recorded rather than assumed.
//!
//! The case is a ring of alternating generator and load buses — sparse, like a
//! real network, and unlike a star it gives no bus a dense Jacobian column.
//! Every generator carries a sixth-order machine with an exciter and a
//! governor, so each unit contributes ten differential states on top of the
//! network's two per bus. A bolted fault is applied and cleared.
//!
//! What to watch is **time per step**, and whether it grows linearly with the
//! system size. It should: the Jacobian is sparse, its pattern is analyzed once
//! for the whole run, and every step is a numeric refactorization against it.
//!
//! Run with `cargo run --release --features dynamics --example dynamics_scale`.

use gridoxide::dynamics::json::{DynamicsData, DynamicsDocument, EventSpec, MachineSpec};
use gridoxide::dynamics::models::machine::GenRoundParams;
use gridoxide::dynamics::models::Limits;
use gridoxide::dynamics::json::{AvrSpec, GovSpec, UnitSpec};
use gridoxide::dynamics::models::avr::SexsParams;
use gridoxide::dynamics::models::gov::Tgov1Params;
use gridoxide::dynamics::{run_dynamics, DynamicsOptions, DynamicsStatus};
use gridoxide::json::NetworkData;
use gridoxide::types::{Bus, BusType, Line};

const S_BASE: f64 = 100.0;

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
fn case(n_bus: usize) -> DynamicsDocument {
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
        // neighbours. A long ring carrying heavy flow is a *power flow*
        // problem, and this example is measuring the dynamics — a base case
        // that fails to solve would put a ceiling here that has nothing to do
        // with what is being timed.
        let p = if generating { 0.30 } else { -0.30 };
        buses.push(bus(i, kind, p, if generating { 0.0 } else { -0.08 }));

        if generating {
            units.push(UnitSpec {
                id: format!("G{i}"),
                bus: i,
                p: None,
                q: None,
                machine: MachineSpec::GenRound(GenRoundParams {
                    h: 5.0,
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
                avr: Some(AvrSpec::Sexs(SexsParams { k: 200.0, ta: 0.1, tb: 1.0, te: 0.05, limits: Limits::NONE })),
                gov: Some(GovSpec::Tgov1(Tgov1Params {
                    r: 0.05,
                    t1: 0.5,
                    t2: 1.0,
                    t3: 5.0,
                    dt: 0.0,
                    limits: Limits::NONE,
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
            events: vec![
                EventSpec::BusFault { t: 1.0, bus: 1, y: None },
                EventSpec::ClearFault { t: 1.08, bus: 1 },
            ],
        },
    }
}

fn main() {
    println!(
        "{:>7}  {:>7}  {:>9}  {:>9}  {:>9}  {:>10}  {:>11}",
        "buses", "units", "unknowns", "build", "run", "per step", "Newton/step"
    );

    for n_bus in [16usize, 64, 256, 1024, 4096] {
        let document = case(n_bus);

        let started = std::time::Instant::now();
        let built = document.build();
        let build_time = started.elapsed();
        let (mut system, events) = match built {
            Ok(pair) => pair,
            Err(e) => {
                println!("{n_bus:>7}  build failed: {e}");
                continue;
            }
        };
        // The invariant still has to hold at scale; a build that drifts here
        // would make every number after it meaningless.
        let drift = system.max_derivative();
        assert!(drift < 1e-9, "{n_bus} buses: initial drift {drift:e}");

        let unknowns = system.n_states() + 2 * system.n_bus();
        let n_units = system.n_states() / 10;

        let options = DynamicsOptions {
            end_time: 3.0,
            step: 0.005,
            events,
            ..Default::default()
        };
        let started = std::time::Instant::now();
        let report = run_dynamics(&mut system, &options);
        let run_time = started.elapsed();
        assert_eq!(report.status, DynamicsStatus::Completed, "{n_bus} buses");

        let per_step = run_time.as_secs_f64() / report.steps as f64;
        println!(
            "{n_bus:>7}  {n_units:>7}  {unknowns:>9}  {:>8.2?}  {:>8.2?}  {:>9.3} ms  {:>11.2}",
            build_time,
            run_time,
            per_step * 1e3,
            report.newton_iterations as f64 / report.steps as f64,
        );
    }

    println!(
        "\nOne symbolic factorization serves each whole run: every event here is value-only, \
         so nothing re-analyzes."
    );
}
