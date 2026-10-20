//! A Dynawo case from its IIDM network: the static network read from the IIDM
//! file, the dynamic models from the `.dyd`/`.par` pair, joined by
//! [`dyd::case_from_network`](crate::dynamics::dyd::case_from_network).

use std::collections::HashMap;
use std::path::Path;

use num_complex::Complex;

use crate::dynamics::DynamicSystem;
use crate::dynamics::dyd::{
    DydError, DydWarning, StaticNetwork, case_from_network, read_dyd, read_par,
};

/// Reads a whole Dynawo case — the IIDM network and the `.dyd`/`.par` pair —
/// and returns a system ready to integrate.
///
/// The IIDM states each generator's id, which is what a `blackBoxModel`'s
/// `staticId` names, and each load's `p0`/`q0`, which is what recovers a
/// machine's own terminal power from its bus's — see
/// [`case_from_network`].
pub fn load_case(
    iidm_path: impl AsRef<Path>,
    dyd_path: impl AsRef<Path>,
    par_path: impl AsRef<Path>,
    f_nom: f64,
) -> Result<(DynamicSystem, Vec<DydWarning>), DydError> {
    let network = super::read(iidm_path).map_err(|e| DydError::Iidm(e.to_string()))?;
    let dyd = read_dyd(dyd_path)?;
    let par = read_par(par_path)?;

    let generators: HashMap<String, usize> = network
        .injections
        .iter()
        .filter(|i| i.generator)
        .map(|i| (i.id.clone(), i.bus))
        .collect();
    let loads = network
        .injections
        .iter()
        .filter(|i| !i.generator)
        .map(|i| (i.bus, Complex::new(i.p, i.q)))
        .collect();

    case_from_network(
        &StaticNetwork {
            buses: &network.buses,
            lines: &network.lines,
            transformers: &network.transformers,
            shunts: &network.shunts,
            base_mva: network.base_mva,
            generators,
            loads,
        },
        &dyd,
        &par,
        f_nom,
    )
}
