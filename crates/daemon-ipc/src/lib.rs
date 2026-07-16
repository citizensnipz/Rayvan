//! Framed JSON-RPC IPC client matching `@rayvan/daemon-contracts` / `@rayvan/daemon-client`.
//!
//! Wire format: u32 little-endian payload length + UTF-8 JSON body.
//! First request must be `system.handshake`.

mod client;
mod framing;
mod paths;

pub use client::{
    event_channel, load_desktop_credential, wait_for_desktop_credential, DaemonConnection,
    DaemonConnectionError, DaemonStatusSnapshot, HandshakeResponse, DESKTOP_CLIENT_ID,
    DESKTOP_CLIENT_TYPE, DESKTOP_CLIENT_VERSION, KEYRING_SERVICE, PROTOCOL_VERSION,
};
pub use framing::{decode_frames, encode_frame, FrameDecoder, FramingError};
pub use paths::{
    daemon_credential_store_path, daemon_endpoint_path, daemon_lock_path, daemon_pid_path,
    default_rayvan_data_dir, default_rayvan_runtime_dir, database_path, user_scope_id,
};
