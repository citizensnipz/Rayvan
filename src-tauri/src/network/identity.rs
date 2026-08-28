use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;
use uuid::Uuid;

pub const NODE_ID_FILE_NAME: &str = "node-id";

#[derive(Debug, Clone)]
pub struct NodeIdentityStore {
    path: PathBuf,
}

impl NodeIdentityStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn load_or_create(&self) -> Result<String, IdentityError> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }

        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&self.path)
        {
            Ok(mut file) => {
                let node_id = Uuid::new_v4().to_string();
                writeln!(file, "{node_id}")?;
                file.sync_all()?;
                Ok(node_id)
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                read_and_validate(&self.path)
            }
            Err(error) => Err(error.into()),
        }
    }
}

fn read_and_validate(path: &Path) -> Result<String, IdentityError> {
    let node_id = fs::read_to_string(path)?.trim().to_owned();
    Uuid::parse_str(&node_id).map_err(|_| IdentityError::InvalidNodeId(path.to_owned()))?;
    Ok(node_id)
}

#[derive(Debug, Error)]
pub enum IdentityError {
    #[error("could not access the local node identity: {0}")]
    Io(#[from] io::Error),
    #[error("the node identity at {0} is not a valid UUID")]
    InvalidNodeId(PathBuf),
}
