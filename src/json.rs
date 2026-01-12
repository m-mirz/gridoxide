use serde::{Deserialize, Serialize};
use super::types::{Bus, Line};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkData {
    pub buses: Vec<Bus>,
    pub lines: Vec<Line>,
}
