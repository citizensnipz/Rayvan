use thiserror::Error;

#[derive(Debug, Error)]
pub enum CredentialStoreError {
    #[error("credential not found")]
    NotFound,
    #[error("credential store unavailable")]
    Unavailable,
    #[error("credential store operation failed: {0}")]
    OperationFailed(String),
}

pub trait CredentialStore {
    fn store_secret(&self, key: &str, value: &str) -> Result<(), CredentialStoreError>;
    fn load_secret(&self, key: &str) -> Result<String, CredentialStoreError>;
    fn delete_secret(&self, key: &str) -> Result<(), CredentialStoreError>;
}

pub struct PlaceholderCredentialStore;

impl CredentialStore for PlaceholderCredentialStore {
    fn store_secret(&self, _key: &str, _value: &str) -> Result<(), CredentialStoreError> {
        Err(CredentialStoreError::Unavailable)
    }

    fn load_secret(&self, _key: &str) -> Result<String, CredentialStoreError> {
        Err(CredentialStoreError::Unavailable)
    }

    fn delete_secret(&self, _key: &str) -> Result<(), CredentialStoreError> {
        Err(CredentialStoreError::Unavailable)
    }
}
