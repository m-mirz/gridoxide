//! PTDF and LODF, checked against the thing they claim to predict.
//!
//! Because DC is *exactly* linear, a finite difference of the DC solve is not
//! an approximation of the derivative — it is the derivative, to round-off.
//! That makes "perturb an injection and re-solve" an exact oracle for PTDF,
//! and "delete a branch and re-solve" an exact oracle for LODF. Neither
//! borrows anything from the sensitivity code, so a shared sign error cannot
//! hide in both.

use std::path::PathBuf;

use gridoxide::linear::{
    dc_branches, dc_power_flow, DcOptions, DcSensitivity,
};
use gridoxide::pgm::pgm_to_buses_and_branches;
use gridoxide::types::{Bus, Line, Transformer};

mod common;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pgm/powerflow")
        .join(rel)
        .join("input.json")
}

struct Case {
    buses: Vec<Bus>,
    lines: Vec<Line>,
    transformers: Vec<Transformer>,
}

impl Case {
    fn load(rel: &str) -> Self {
        let input = common::load_pgm_input(&fixture(rel));
        let (buses, lines, transformers) = pgm_to_buses_and_branches(input, 1e6, 50.0);
        Self { buses, lines, transformers }
    }

    fn n_branches(&self) -> usize {
        self.lines.len() + self.transformers.len()
    }

    /// Solves DC with each bus's injection offset by `delta`, returning the
    /// per-branch flows. Perturbing `p_spec` and leaving the slack to absorb
    /// it is exactly the transfer PTDF describes.
    fn flows_with(&self, delta: &[f64]) -> Vec<f64> {
        let mut buses = self.buses.clone();
        for (b, d) in buses.iter_mut().zip(delta) {
            b.p_spec += d;
        }
        dc_power_flow(&mut buses, &self.lines, &self.transformers, DcOptions::default()).branch_p
    }

    fn sensitivity(&self) -> DcSensitivity {
        let branches = dc_branches(&self.lines, &self.transformers, DcOptions::default());
        DcSensitivity::new(&self.buses, &branches, self.n_branches())
            .expect("every island here has a reference and a nonsingular B")
    }
}

/// PTDF against a finite difference of the DC solve itself.
///
/// The step size is irrelevant to the answer — DC is linear, so any step
/// recovers the same derivative — and `1e-3` is chosen only to stay well clear
/// of cancellation in the subtraction.
#[test]
fn ptdf_columns_match_a_finite_difference_of_the_solve() {
    for name in ["symmetric/transmission-case", "symmetric/distribution-case"] {
        let case = Case::load(name);
        let sensitivity = case.sensitivity();
        let base = case.flows_with(&vec![0.0; case.buses.len()]);
        let eps = 1e-3;

        let mut checked = 0;
        for bus in 0..case.buses.len() {
            let Some(column) = sensitivity.ptdf_column(bus) else { continue };

            let mut delta = vec![0.0; case.buses.len()];
            delta[bus] = eps;
            let perturbed = case.flows_with(&delta);

            for k in 0..case.n_branches() {
                let measured = (perturbed[k] - base[k]) / eps;
                assert!(
                    (measured - column[k]).abs() < 1e-9,
                    "{name}: PTDF[{k}, {bus}] = {} but re-solving gives {measured}",
                    column[k]
                );
            }
            checked += 1;
        }
        assert!(checked > 2, "{name}: only {checked} buses had PTDF columns");
    }
}

/// `ptdf_row` takes one solve where `ptdf_column` would take `n_buses`. It
/// must agree with them exactly, since the shortcut rests on `B_NN` being
/// symmetric — which holds only because the phase shift lives on the
/// right-hand side.
#[test]
fn ptdf_rows_agree_with_ptdf_columns() {
    let case = Case::load("symmetric/transmission-case");
    let sensitivity = case.sensitivity();

    let columns: Vec<Option<Vec<f64>>> =
        (0..case.buses.len()).map(|b| sensitivity.ptdf_column(b)).collect();

    let mut checked = 0;
    for branch in 0..case.n_branches() {
        let Some(row) = sensitivity.ptdf_row(branch) else { continue };
        for (bus, column) in columns.iter().enumerate() {
            let Some(column) = column else { continue };
            assert!(
                (row[bus] - column[branch]).abs() < 1e-12,
                "branch {branch}, bus {bus}: row {} vs column {}",
                row[bus],
                column[branch]
            );
        }
        checked += 1;
    }
    assert!(checked > 2, "only {checked} branches had PTDF rows");
}

/// LODF against an actual outage: delete the branch, re-solve, and check that
/// the flow each surviving branch picked up is the predicted fraction of what
/// the outaged branch had been carrying.
///
/// Deleting a `Line` is done by opening it into gridoxide's own half-open
/// representation (`from == to`), which `dc_branches` skips — the same path a
/// genuinely open branch takes, rather than a special case invented for this
/// test.
#[test]
fn lodf_columns_match_an_actual_outage_resolve() {
    let case = Case::load("symmetric/transmission-case");
    let sensitivity = case.sensitivity();
    let base = case.flows_with(&vec![0.0; case.buses.len()]);

    let mut checked = 0;
    for outaged in 0..case.lines.len() {
        let Some(column) = sensitivity.lodf_column(outaged) else { continue };

        // Re-solve without that line.
        let mut lines = case.lines.clone();
        lines[outaged].to = lines[outaged].from;
        let mut buses = case.buses.clone();
        let after =
            dc_power_flow(&mut buses, &lines, &case.transformers, DcOptions::default()).branch_p;

        let lost = base[outaged];
        if lost.abs() < 1e-9 {
            // A branch carrying nothing redistributes nothing; the identity
            // holds trivially and proves nothing, so skip it.
            continue;
        }

        for k in 0..case.n_branches() {
            if k == outaged {
                continue;
            }
            let predicted = base[k] + column[k] * lost;
            assert!(
                (after[k] - predicted).abs() < 1e-9,
                "outage of {outaged}: branch {k} went to {} but LODF predicted {predicted}",
                after[k]
            );
        }
        assert_eq!(column[outaged], -1.0);
        checked += 1;
    }
    assert!(checked > 0, "no non-radial, load-carrying branch was checked");
}

/// A radial branch has no LODF column, because removing it islands the network
/// rather than rerouting anything — and the fixture must actually contain one,
/// or `is_radial` is never exercised on real data.
#[test]
fn radial_branches_are_identified_and_have_no_factors() {
    let case = Case::load("symmetric/transmission-case");
    let sensitivity = case.sensitivity();

    let mut radial = 0;
    let mut meshed = 0;
    for branch in 0..case.n_branches() {
        if sensitivity.is_radial(branch) {
            assert!(
                sensitivity.lodf_column(branch).is_none(),
                "branch {branch} is radial but produced an LODF column"
            );
            radial += 1;
        } else {
            assert!(sensitivity.lodf_column(branch).is_some());
            meshed += 1;
        }
    }
    assert!(radial > 0, "fixture has no radial branch, so is_radial is untested here");
    assert!(meshed > 0, "fixture has no meshed branch, so lodf_column is untested here");
}

/// `transfer_factors` is the primitive the columns are special cases of, so a
/// multi-bus transfer must match the same re-solve oracle PTDF is held to.
#[test]
fn transfer_factors_match_a_multi_bus_re_solve() {
    let case = Case::load("symmetric/transmission-case");
    let sensitivity = case.sensitivity();
    let base = case.flows_with(&vec![0.0; case.buses.len()]);

    // An arbitrary spread-out pattern; the slack absorbs the imbalance.
    let mut delta = vec![0.0; case.buses.len()];
    for (i, d) in delta.iter_mut().enumerate() {
        if case.buses[i].bus_type != gridoxide::types::BusType::Slack {
            *d = 0.01 * ((i % 3) as f64 - 1.0);
        }
    }

    let predicted = sensitivity.transfer_factors(&delta).unwrap();
    let actual = case.flows_with(&delta);
    for k in 0..case.n_branches() {
        assert!(
            (actual[k] - base[k] - predicted[k]).abs() < 1e-9,
            "branch {k}: measured {} but transfer_factors predicted {}",
            actual[k] - base[k],
            predicted[k]
        );
    }
}

/// The dense matrices are the accessors laid out in memory, nothing more.
/// Checked on a small fixture — at `case9241pegase` scale these would be
/// 1.19 GB and 2.06 GB, which is why the accessors are the real API.
#[test]
fn dense_matrices_agree_with_the_accessors() {
    let case = Case::load("symmetric/transmission-case");
    let sensitivity = case.sensitivity();

    let ptdf = sensitivity.ptdf_dense().unwrap();
    assert_eq!((ptdf.rows, ptdf.cols), (case.n_branches(), case.buses.len()));
    for bus in 0..case.buses.len() {
        let Some(column) = sensitivity.ptdf_column(bus) else { continue };
        for branch in 0..case.n_branches() {
            assert_eq!(ptdf.get(branch, bus), column[branch]);
        }
    }

    let lodf = sensitivity.lodf_dense().unwrap();
    assert_eq!((lodf.rows, lodf.cols), (case.n_branches(), case.n_branches()));
    for l in 0..case.n_branches() {
        let Some(column) = sensitivity.lodf_column(l) else { continue };
        for k in 0..case.n_branches() {
            assert_eq!(lodf.get(k, l), column[k]);
        }
    }
}
