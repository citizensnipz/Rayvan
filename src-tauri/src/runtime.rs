use crate::domain::{Activation, Shard};
use std::error::Error;
use std::fmt;

pub trait ShardRuntime {
    type Error: Error;

    fn load_shard(&mut self, shard: &Shard) -> Result<(), Self::Error>;
    fn forward(&mut self, activation: Activation) -> Result<Activation, Self::Error>;
}

#[derive(Debug, Default)]
pub struct DummyRuntime {
    loaded_shard: Option<Shard>,
}

impl DummyRuntime {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ShardRuntime for DummyRuntime {
    type Error = DummyRuntimeError;

    fn load_shard(&mut self, shard: &Shard) -> Result<(), Self::Error> {
        self.loaded_shard = Some(shard.clone());
        Ok(())
    }

    fn forward(&mut self, mut activation: Activation) -> Result<Activation, Self::Error> {
        let shard = self
            .loaded_shard
            .as_ref()
            .ok_or(DummyRuntimeError::ShardNotLoaded)?;

        if activation.model_id != shard.model_id {
            return Err(DummyRuntimeError::ModelMismatch);
        }
        if activation.next_layer != shard.start_layer {
            return Err(DummyRuntimeError::UnexpectedLayer {
                expected: shard.start_layer,
                actual: activation.next_layer,
            });
        }

        activation.payload.extend_from_slice(shard.id.as_bytes());
        activation.next_layer = shard.end_layer;
        Ok(activation)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DummyRuntimeError {
    ShardNotLoaded,
    ModelMismatch,
    UnexpectedLayer { expected: u32, actual: u32 },
}

impl fmt::Display for DummyRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShardNotLoaded => write!(formatter, "no shard is loaded"),
            Self::ModelMismatch => write!(formatter, "activation and shard models do not match"),
            Self::UnexpectedLayer { expected, actual } => write!(
                formatter,
                "activation starts at layer {actual}, but the shard starts at layer {expected}"
            ),
        }
    }
}

impl Error for DummyRuntimeError {}
