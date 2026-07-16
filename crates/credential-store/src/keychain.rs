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

/// OS keyring-backed store compatible with `@napi-rs/keyring` / `@rayvan/daemon-client`.
///
/// Service defaults to `com.rayvan.local-client`; account is the client id
/// (e.g. `rayvan-desktop`).
pub struct OsKeyringCredentialStore {
    service: String,
}

impl OsKeyringCredentialStore {
    pub fn new(service: impl Into<String>) -> Self {
        Self {
            service: service.into(),
        }
    }

    pub fn rayvan_local_clients() -> Self {
        Self::new("com.rayvan.local-client")
    }

    fn entry(&self, account: &str) -> Result<keyring::Entry, CredentialStoreError> {
        keyring::Entry::new(&self.service, account).map_err(|error| {
            CredentialStoreError::OperationFailed(format!("keyring entry create failed: {error}"))
        })
    }
}

impl Default for OsKeyringCredentialStore {
    fn default() -> Self {
        Self::rayvan_local_clients()
    }
}

impl CredentialStore for OsKeyringCredentialStore {
    fn store_secret(&self, key: &str, value: &str) -> Result<(), CredentialStoreError> {
        self.entry(key)?
            .set_password(value)
            .map_err(|error| CredentialStoreError::OperationFailed(error.to_string()))
    }

    fn load_secret(&self, key: &str) -> Result<String, CredentialStoreError> {
        match self.entry(key)?.get_password() {
            Ok(value) => Ok(value),
            Err(keyring::Error::NoEntry) => Err(CredentialStoreError::NotFound),
            Err(error) => {
                let message = error.to_string();
                if is_missing_credential(&message) {
                    Err(CredentialStoreError::NotFound)
                } else {
                    Err(CredentialStoreError::OperationFailed(message))
                }
            }
        }
    }

    fn delete_secret(&self, key: &str) -> Result<(), CredentialStoreError> {
        match self.entry(key)?.delete_credential() {
            Ok(()) => Ok(()),
            Err(keyring::Error::NoEntry) => Ok(()),
            Err(error) => {
                let message = error.to_string();
                if is_missing_credential(&message) {
                    Ok(())
                } else {
                    Err(CredentialStoreError::OperationFailed(message))
                }
            }
        }
    }
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

fn is_missing_credential(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("no entry")
        || lower.contains("not found")
        || lower.contains("does not exist")
}
