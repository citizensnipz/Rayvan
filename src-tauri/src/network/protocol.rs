use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::io;
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub const PROTOCOL_VERSION: u16 = 1;
const MAX_FRAME_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum MembershipMessage {
    RegisterNode {
        protocol_version: u16,
        node_id: String,
    },
    JoinAccepted {
        protocol_version: u16,
    },
    JoinRejected {
        reason: String,
    },
    Heartbeat {
        node_id: String,
    },
}

pub(crate) async fn write_message<W>(
    writer: &mut W,
    message: &MembershipMessage,
) -> Result<(), FrameError>
where
    W: AsyncWrite + Unpin,
{
    let payload = serde_json::to_vec(message)?;
    if payload.len() > MAX_FRAME_BYTES {
        return Err(FrameError::FrameTooLarge(payload.len()));
    }

    writer
        .write_all(&(payload.len() as u32).to_be_bytes())
        .await?;
    writer.write_all(&payload).await?;
    writer.flush().await?;
    Ok(())
}

pub(crate) async fn read_message<R>(reader: &mut R) -> Result<MembershipMessage, FrameError>
where
    R: AsyncRead + Unpin,
{
    read_frame(reader).await
}

async fn read_frame<R, T>(reader: &mut R) -> Result<T, FrameError>
where
    R: AsyncRead + Unpin,
    T: DeserializeOwned,
{
    let mut length_bytes = [0_u8; 4];
    reader.read_exact(&mut length_bytes).await?;
    let length = u32::from_be_bytes(length_bytes) as usize;
    if length > MAX_FRAME_BYTES {
        return Err(FrameError::FrameTooLarge(length));
    }

    let mut payload = vec![0_u8; length];
    reader.read_exact(&mut payload).await?;
    Ok(serde_json::from_slice(&payload)?)
}

#[derive(Debug, Error)]
pub(crate) enum FrameError {
    #[error("membership frame I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("membership frame is {0} bytes, exceeding the 16 KiB limit")]
    FrameTooLarge(usize),
    #[error("membership frame is not valid JSON: {0}")]
    Json(#[from] serde_json::Error),
}
