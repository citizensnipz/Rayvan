use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use interprocess::local_socket::prelude::*;
use interprocess::local_socket::Stream as LocalSocketStream;
use interprocess::TryClone;
use rayvan_credential_store::{CredentialStore, CredentialStoreError, OsKeyringCredentialStore};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;

use crate::framing::{encode_frame, FrameDecoder, FramingError};
use crate::paths::daemon_endpoint_path;

pub const PROTOCOL_VERSION: &str = "1";
pub const DESKTOP_CLIENT_ID: &str = "rayvan-desktop";
pub const DESKTOP_CLIENT_TYPE: &str = "desktop";
pub const DESKTOP_CLIENT_VERSION: &str = "0.0.1";
pub const KEYRING_SERVICE: &str = "com.rayvan.local-client";

#[derive(Debug, Error)]
pub enum DaemonConnectionError {
    #[error("daemon unavailable: {0}")]
    Unavailable(String),
    #[error("authentication failed: {0}")]
    Unauthenticated(String),
    #[error("credential store error: {0}")]
    Credential(#[from] CredentialStoreError),
    #[error("framing error: {0}")]
    Framing(#[from] FramingError),
    #[error("request timed out: {0}")]
    Timeout(String),
    #[error("daemon error {code}: {message}")]
    Rpc {
        code: String,
        message: String,
        data: Option<Value>,
    },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("internal error: {0}")]
    Internal(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HandshakeResponse {
    pub protocol_version: String,
    pub daemon_version: String,
    pub session_id: String,
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub authenticated_client_id: Option<String>,
    #[serde(default)]
    pub permission_profile_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DaemonStatusSnapshot {
    pub connected: bool,
    pub endpoint: String,
    pub spawned: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub daemon_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authenticated_client_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<Value>,
}

type PendingMap = HashMap<String, Sender<Result<Value, DaemonConnectionError>>>;

struct ConnectionInner {
    writer: Mutex<LocalSocketStream>,
    pending: Mutex<PendingMap>,
    event_tx: Mutex<Option<Sender<Value>>>,
    request_counter: AtomicU64,
    handshake: Mutex<Option<HandshakeResponse>>,
}

pub struct DaemonConnection {
    inner: Arc<ConnectionInner>,
    endpoint: String,
}

impl DaemonConnection {
    pub fn connect(
        endpoint: Option<String>,
        client_credential: Option<String>,
        connect_timeout: Duration,
        request_timeout: Duration,
    ) -> Result<(Self, HandshakeResponse), DaemonConnectionError> {
        let endpoint = endpoint.unwrap_or_else(|| daemon_endpoint_path(None));
        let stream = open_stream(&endpoint, connect_timeout)?;
        let reader = stream
            .try_clone()
            .map_err(|error| DaemonConnectionError::Io(error))?;
        let inner = Arc::new(ConnectionInner {
            writer: Mutex::new(stream),
            pending: Mutex::new(HashMap::new()),
            event_tx: Mutex::new(None),
            request_counter: AtomicU64::new(0),
            handshake: Mutex::new(None),
        });

        let conn = Self {
            inner: Arc::clone(&inner),
            endpoint,
        };
        conn.spawn_reader(reader);

        let credential = match client_credential {
            Some(value) => value,
            None => load_desktop_credential()?,
        };

        let handshake_params = json!({
            "protocolVersion": PROTOCOL_VERSION,
            "clientType": DESKTOP_CLIENT_TYPE,
            "clientVersion": DESKTOP_CLIENT_VERSION,
            "clientId": DESKTOP_CLIENT_ID,
            "clientCredential": credential,
        });

        let result =
            conn.request_with_timeout("system.handshake", handshake_params, request_timeout)?;
        let handshake: HandshakeResponse = serde_json::from_value(result)
            .map_err(|error| DaemonConnectionError::Internal(error.to_string()))?;
        *conn.inner.handshake.lock().unwrap() = Some(handshake.clone());
        Ok((conn, handshake))
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn handshake(&self) -> Option<HandshakeResponse> {
        self.inner.handshake.lock().unwrap().clone()
    }

    pub fn set_event_sender(&self, sender: Sender<Value>) {
        *self.inner.event_tx.lock().unwrap() = Some(sender);
    }

    pub fn request(&self, method: &str, params: Value) -> Result<Value, DaemonConnectionError> {
        self.request_with_timeout(method, params, Duration::from_secs(30))
    }

    pub fn request_with_timeout(
        &self,
        method: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value, DaemonConnectionError> {
        let id = format!(
            "req_{}_{}",
            self.inner.request_counter.fetch_add(1, Ordering::Relaxed) + 1,
            now_millis()
        );
        let envelope = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        let frame = encode_frame(&envelope)?;
        let (tx, rx) = mpsc::channel();
        {
            let mut pending = self.inner.pending.lock().unwrap();
            pending.insert(id.clone(), tx);
        }
        {
            let mut stream = self.inner.writer.lock().unwrap();
            if let Err(error) = stream.write_all(&frame) {
                self.inner.pending.lock().unwrap().remove(&id);
                return Err(DaemonConnectionError::Io(error));
            }
            let _ = stream.flush();
        }

        match rx.recv_timeout(timeout) {
            Ok(result) => result,
            Err(_) => {
                self.inner.pending.lock().unwrap().remove(&id);
                Err(DaemonConnectionError::Timeout(method.to_string()))
            }
        }
    }

    pub fn subscribe_all(&self) -> Result<(), DaemonConnectionError> {
        self.request("system.subscribe", json!({ "eventTypes": ["*"] }))?;
        Ok(())
    }

    fn spawn_reader(&self, mut reader: LocalSocketStream) {
        let inner = Arc::clone(&self.inner);
        thread::spawn(move || {
            let mut buffer = [0u8; 8192];
            let mut decoder = FrameDecoder::new();
            loop {
                match reader.read(&mut buffer) {
                    Ok(0) => {
                        fail_all(
                            &inner,
                            DaemonConnectionError::Unavailable("connection closed".into()),
                        );
                        break;
                    }
                    Ok(n) => match decoder.push(&buffer[..n]) {
                        Ok(messages) => {
                            for message in messages {
                                handle_message(&inner, message);
                            }
                        }
                        Err(error) => {
                            fail_all(&inner, DaemonConnectionError::Framing(error));
                            break;
                        }
                    },
                    Err(error) => {
                        fail_all(&inner, DaemonConnectionError::Io(error));
                        break;
                    }
                }
            }
        });
    }
}

fn handle_message(inner: &ConnectionInner, message: Value) {
    if message.get("method").and_then(|v| v.as_str()) == Some("daemon.event") {
        if let Some(params) = message.get("params").cloned() {
            if let Some(tx) = inner.event_tx.lock().unwrap().as_ref() {
                let _ = tx.send(params);
            }
        }
        return;
    }

    let Some(id) = message
        .get("id")
        .and_then(|v| v.as_str().map(str::to_string))
    else {
        return;
    };
    let pending = inner.pending.lock().unwrap().remove(&id);
    let Some(tx) = pending else {
        return;
    };

    if let Some(error) = message.get("error") {
        let code = error
            .pointer("/data/code")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .or_else(|| error.get("code").map(|v| v.to_string()))
            .unwrap_or_else(|| "INTERNAL_ERROR".into());
        let message_text = error
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("Daemon request failed")
            .to_string();
        let data = error.get("data").cloned();
        let _ = tx.send(Err(DaemonConnectionError::Rpc {
            code,
            message: message_text,
            data,
        }));
        return;
    }

    let result = message.get("result").cloned().unwrap_or(Value::Null);
    let _ = tx.send(Ok(result));
}

fn fail_all(inner: &ConnectionInner, error: DaemonConnectionError) {
    let mut pending = inner.pending.lock().unwrap();
    for (_, tx) in pending.drain() {
        let _ = tx.send(Err(DaemonConnectionError::Unavailable(error.to_string())));
    }
}

fn open_stream(
    endpoint: &str,
    timeout: Duration,
) -> Result<LocalSocketStream, DaemonConnectionError> {
    let started = Instant::now();
    loop {
        match try_connect(endpoint) {
            Ok(stream) => return Ok(stream),
            Err(error) if started.elapsed() >= timeout => {
                return Err(DaemonConnectionError::Unavailable(format!(
                    "timed out connecting to daemon at {endpoint}: {error}"
                )));
            }
            Err(_) => thread::sleep(Duration::from_millis(50)),
        }
    }
}

#[cfg(windows)]
fn try_connect(endpoint: &str) -> Result<LocalSocketStream, DaemonConnectionError> {
    use interprocess::local_socket::{GenericNamespaced, ToNsName};

    let name = endpoint
        .trim_start_matches(r"\\.\pipe\")
        .to_ns_name::<GenericNamespaced>()
        .map_err(|error| DaemonConnectionError::Unavailable(error.to_string()))?;
    LocalSocketStream::connect(name)
        .map_err(|error| DaemonConnectionError::Unavailable(error.to_string()))
}

#[cfg(unix)]
fn try_connect(endpoint: &str) -> Result<LocalSocketStream, DaemonConnectionError> {
    use interprocess::local_socket::{GenericFilePath, ToFsName};

    let name = endpoint
        .to_fs_name::<GenericFilePath>()
        .map_err(|error| DaemonConnectionError::Unavailable(error.to_string()))?;
    LocalSocketStream::connect(name)
        .map_err(|error| DaemonConnectionError::Unavailable(error.to_string()))
}

#[cfg(not(any(windows, unix)))]
fn try_connect(_endpoint: &str) -> Result<LocalSocketStream, DaemonConnectionError> {
    Err(DaemonConnectionError::Unavailable(
        "local IPC unsupported on this platform".into(),
    ))
}

pub fn load_desktop_credential() -> Result<String, DaemonConnectionError> {
    let store = OsKeyringCredentialStore::new(KEYRING_SERVICE);
    match store.load_secret(DESKTOP_CLIENT_ID) {
        Ok(value) => Ok(value),
        Err(CredentialStoreError::NotFound) => Err(DaemonConnectionError::Unauthenticated(
            format!(
                "credential for {DESKTOP_CLIENT_ID} not found in OS keyring (service {KEYRING_SERVICE})"
            ),
        )),
        Err(error) => Err(DaemonConnectionError::Credential(error)),
    }
}

pub fn wait_for_desktop_credential(timeout: Duration) -> Result<String, DaemonConnectionError> {
    let started = Instant::now();
    loop {
        match load_desktop_credential() {
            Ok(value) => return Ok(value),
            Err(DaemonConnectionError::Unauthenticated(_)) if started.elapsed() < timeout => {
                thread::sleep(Duration::from_millis(100));
            }
            Err(error) => return Err(error),
        }
    }
}

fn now_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// Event listener helper: create a channel and attach it to the connection.
pub fn event_channel() -> (Sender<Value>, Receiver<Value>) {
    mpsc::channel()
}
