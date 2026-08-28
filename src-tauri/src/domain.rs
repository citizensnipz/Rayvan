use std::collections::HashSet;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    Available,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeCapabilities {
    pub available_memory_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub id: String,
    pub status: NodeStatus,
    pub capabilities: NodeCapabilities,
}

impl Node {
    pub fn new(id: impl Into<String>, status: NodeStatus, capabilities: NodeCapabilities) -> Self {
        Self {
            id: id.into(),
            status,
            capabilities,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelManifest {
    pub model_id: String,
    pub total_layers: u32,
    pub revision: Option<String>,
}

impl ModelManifest {
    pub fn new(model_id: impl Into<String>, total_layers: u32, revision: Option<String>) -> Self {
        Self {
            model_id: model_id.into(),
            total_layers,
            revision,
        }
    }
}

/// A contiguous, half-open transformer-layer range: `[start_layer, end_layer)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shard {
    pub id: String,
    pub model_id: String,
    pub start_layer: u32,
    pub end_layer: u32,
    pub assigned_node_id: String,
}

impl Shard {
    pub fn new(
        id: impl Into<String>,
        model_id: impl Into<String>,
        start_layer: u32,
        end_layer: u32,
        assigned_node_id: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            model_id: model_id.into(),
            start_layer,
            end_layer,
            assigned_node_id: assigned_node_id.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Activation {
    pub model_id: String,
    pub request_id: String,
    pub next_layer: u32,
    pub payload: Vec<u8>,
}

impl Activation {
    pub fn new(
        model_id: impl Into<String>,
        request_id: impl Into<String>,
        next_layer: u32,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            request_id: request_id.into(),
            next_layer,
            payload,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Swarm {
    manifest: ModelManifest,
    nodes: Vec<Node>,
    shards: Vec<Shard>,
}

impl Swarm {
    pub fn new(
        manifest: ModelManifest,
        nodes: Vec<Node>,
        mut shards: Vec<Shard>,
    ) -> Result<Self, SwarmValidationError> {
        validate_nodes(&nodes)?;
        shards.sort_unstable_by_key(|shard| (shard.start_layer, shard.end_layer));
        validate_shards(&manifest, &nodes, &shards)?;

        Ok(Self {
            manifest,
            nodes,
            shards,
        })
    }

    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn shards(&self) -> &[Shard] {
        &self.shards
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwarmValidationError {
    EmptyModel,
    DuplicateNode { node_id: String },
    UnknownNode { shard_id: String, node_id: String },
    ModelMismatch { shard_id: String },
    InvalidRange { shard_id: String },
    Gap { start_layer: u32, end_layer: u32 },
    Overlap { start_layer: u32, end_layer: u32 },
}

impl fmt::Display for SwarmValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyModel => write!(formatter, "the model manifest has no transformer layers"),
            Self::DuplicateNode { node_id } => {
                write!(formatter, "node {node_id} appears more than once")
            }
            Self::UnknownNode { shard_id, node_id } => {
                write!(
                    formatter,
                    "shard {shard_id} is assigned to unknown node {node_id}"
                )
            }
            Self::ModelMismatch { shard_id } => {
                write!(formatter, "shard {shard_id} belongs to a different model")
            }
            Self::InvalidRange { shard_id } => {
                write!(formatter, "shard {shard_id} has an invalid layer range")
            }
            Self::Gap {
                start_layer,
                end_layer,
            } => write!(
                formatter,
                "layers {start_layer}..{end_layer} are unassigned"
            ),
            Self::Overlap {
                start_layer,
                end_layer,
            } => write!(formatter, "layers {start_layer}..{end_layer} overlap"),
        }
    }
}

impl Error for SwarmValidationError {}

fn validate_nodes(nodes: &[Node]) -> Result<(), SwarmValidationError> {
    let mut node_ids = HashSet::with_capacity(nodes.len());
    for node in nodes {
        if !node_ids.insert(node.id.as_str()) {
            return Err(SwarmValidationError::DuplicateNode {
                node_id: node.id.clone(),
            });
        }
    }
    Ok(())
}

fn validate_shards(
    manifest: &ModelManifest,
    nodes: &[Node],
    shards: &[Shard],
) -> Result<(), SwarmValidationError> {
    if manifest.total_layers == 0 {
        return Err(SwarmValidationError::EmptyModel);
    }

    let mut next_layer = 0;
    for shard in shards {
        if shard.model_id != manifest.model_id {
            return Err(SwarmValidationError::ModelMismatch {
                shard_id: shard.id.clone(),
            });
        }
        if !nodes.iter().any(|node| node.id == shard.assigned_node_id) {
            return Err(SwarmValidationError::UnknownNode {
                shard_id: shard.id.clone(),
                node_id: shard.assigned_node_id.clone(),
            });
        }
        if shard.start_layer >= shard.end_layer || shard.end_layer > manifest.total_layers {
            return Err(SwarmValidationError::InvalidRange {
                shard_id: shard.id.clone(),
            });
        }
        if shard.start_layer > next_layer {
            return Err(SwarmValidationError::Gap {
                start_layer: next_layer,
                end_layer: shard.start_layer,
            });
        }
        if shard.start_layer < next_layer {
            return Err(SwarmValidationError::Overlap {
                start_layer: shard.start_layer,
                end_layer: next_layer.min(shard.end_layer),
            });
        }
        next_layer = shard.end_layer;
    }

    if next_layer < manifest.total_layers {
        return Err(SwarmValidationError::Gap {
            start_layer: next_layer,
            end_layer: manifest.total_layers,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_ID: &str = "rayvan/test-model";

    fn node(id: &str) -> Node {
        Node::new(
            id,
            NodeStatus::Available,
            NodeCapabilities {
                available_memory_bytes: 8 * 1024 * 1024 * 1024,
            },
        )
    }

    fn shard(id: &str, start_layer: u32, end_layer: u32, node_id: &str) -> Shard {
        Shard::new(id, MODEL_ID, start_layer, end_layer, node_id)
    }

    #[test]
    fn constructs_valid_swarm() {
        let swarm = Swarm::new(
            ModelManifest::new(MODEL_ID, 6, Some("test-revision".to_owned())),
            vec![node("node-a"), node("node-b"), node("node-c")],
            vec![
                shard("shard-c", 4, 6, "node-c"),
                shard("shard-a", 0, 2, "node-a"),
                shard("shard-b", 2, 4, "node-b"),
            ],
        )
        .expect("the shards cover the model exactly");

        let starts: Vec<_> = swarm
            .shards()
            .iter()
            .map(|shard| shard.start_layer)
            .collect();
        assert_eq!(starts, vec![0, 2, 4]);
        assert_eq!(swarm.manifest().total_layers, 6);
        assert_eq!(swarm.nodes().len(), 3);
    }

    #[test]
    fn detects_missing_layers() {
        let error = Swarm::new(
            ModelManifest::new(MODEL_ID, 6, None),
            vec![node("node-a"), node("node-b")],
            vec![
                shard("shard-a", 0, 2, "node-a"),
                shard("shard-b", 3, 6, "node-b"),
            ],
        )
        .expect_err("layer 2 is missing");

        assert_eq!(
            error,
            SwarmValidationError::Gap {
                start_layer: 2,
                end_layer: 3,
            }
        );
    }

    #[test]
    fn detects_overlapping_shards() {
        let error = Swarm::new(
            ModelManifest::new(MODEL_ID, 6, None),
            vec![node("node-a"), node("node-b")],
            vec![
                shard("shard-a", 0, 4, "node-a"),
                shard("shard-b", 3, 6, "node-b"),
            ],
        )
        .expect_err("layer 3 is assigned twice");

        assert_eq!(
            error,
            SwarmValidationError::Overlap {
                start_layer: 3,
                end_layer: 4,
            }
        );
    }
}
