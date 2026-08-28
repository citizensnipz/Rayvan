use crate::domain::{
    Activation, ModelManifest, Node, NodeCapabilities, NodeStatus, Shard, Swarm,
    SwarmValidationError,
};
use crate::runtime::{DummyRuntime, DummyRuntimeError, ShardRuntime};
use std::error::Error;
use std::fmt;

const MODEL_ID: &str = "rayvan/dummy-model";

pub fn run_three_node_simulation() -> Result<Activation, SimulationError> {
    let capabilities = NodeCapabilities {
        available_memory_bytes: 8 * 1024 * 1024 * 1024,
    };
    let nodes = ["node-a", "node-b", "node-c"]
        .into_iter()
        .map(|id| Node::new(id, NodeStatus::Available, capabilities))
        .collect();
    let shards = vec![
        Shard::new("shard-0", MODEL_ID, 0, 1, "node-a"),
        Shard::new("shard-1", MODEL_ID, 1, 2, "node-b"),
        Shard::new("shard-2", MODEL_ID, 2, 3, "node-c"),
    ];
    let swarm = Swarm::new(ModelManifest::new(MODEL_ID, 3, None), nodes, shards)?;

    let mut runtimes = Vec::with_capacity(swarm.shards().len());
    for shard in swarm.shards() {
        let mut runtime = DummyRuntime::new();
        runtime.load_shard(shard)?;
        runtimes.push(runtime);
    }

    let mut activation = Activation::new(MODEL_ID, "local-request", 0, Vec::new());
    for runtime in &mut runtimes {
        activation = runtime.forward(activation)?;
    }

    Ok(activation)
}

#[derive(Debug)]
pub enum SimulationError {
    InvalidSwarm(SwarmValidationError),
    Runtime(DummyRuntimeError),
}

impl fmt::Display for SimulationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSwarm(error) => write!(formatter, "invalid local swarm: {error}"),
            Self::Runtime(error) => write!(formatter, "dummy runtime failed: {error}"),
        }
    }
}

impl Error for SimulationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidSwarm(error) => Some(error),
            Self::Runtime(error) => Some(error),
        }
    }
}

impl From<SwarmValidationError> for SimulationError {
    fn from(error: SwarmValidationError) -> Self {
        Self::InvalidSwarm(error)
    }
}

impl From<DummyRuntimeError> for SimulationError {
    fn from(error: DummyRuntimeError) -> Self {
        Self::Runtime(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_passes_through_three_shards_in_order() {
        let activation =
            run_three_node_simulation().expect("the local three-node pipeline should complete");

        assert_eq!(activation.model_id, MODEL_ID);
        assert_eq!(activation.request_id, "local-request");
        assert_eq!(activation.next_layer, 3);
        assert_eq!(activation.payload, b"shard-0shard-1shard-2");
    }
}
