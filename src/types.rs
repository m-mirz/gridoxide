#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BusType {
    Slack,
    PV,
    PQ,
}

#[derive(Clone, Debug)]
pub struct Bus {
    pub idx: usize,          // index in arrays (0-based)
    pub bus_type: BusType,
    pub voltage_mag: f64,    // Vm (p.u.)
    pub voltage_ang: f64,    // Va (rad)
    pub p_spec: f64,         // P specified (generation - load) in p.u.
    pub q_spec: f64,         // Q specified (generation - load) in p.u.
    pub q_min: f64,          // reactive limits (for PV handling, optional)
    pub q_max: f64,
}

#[derive(Clone, Debug)]
pub struct Line {
    pub from: usize,
    pub to: usize,
    pub r: f64,
    pub x: f64,
    pub b_shunt: f64, // total line charging
}
