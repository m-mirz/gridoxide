use super::types::{Bus, Line};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkData {
    pub buses: Vec<Bus>,
    pub lines: Vec<Line>,
}
