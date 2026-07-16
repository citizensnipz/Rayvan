use thiserror::Error;

const MAX_FRAME_BYTES: u32 = 16 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum FramingError {
    #[error("frame length {0} exceeds maximum")]
    FrameTooLarge(u32),
    #[error("invalid UTF-8 in frame body")]
    InvalidUtf8(#[from] std::str::Utf8Error),
    #[error("invalid JSON in frame body: {0}")]
    InvalidJson(#[from] serde_json::Error),
}

pub fn encode_frame(payload: &serde_json::Value) -> Result<Vec<u8>, FramingError> {
    let body = serde_json::to_vec(payload)?;
    let length = u32::try_from(body.len()).map_err(|_| FramingError::FrameTooLarge(u32::MAX))?;
    if length > MAX_FRAME_BYTES {
        return Err(FramingError::FrameTooLarge(length));
    }
    let mut out = Vec::with_capacity(4 + body.len());
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

pub struct FrameDecoder {
    buffer: Vec<u8>,
}

impl Default for FrameDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameDecoder {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
        }
    }

    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<serde_json::Value>, FramingError> {
        self.buffer.extend_from_slice(chunk);
        let mut messages = Vec::new();

        while self.buffer.len() >= 4 {
            let length = u32::from_le_bytes([
                self.buffer[0],
                self.buffer[1],
                self.buffer[2],
                self.buffer[3],
            ]);
            if length > MAX_FRAME_BYTES {
                return Err(FramingError::FrameTooLarge(length));
            }
            let total = 4 + length as usize;
            if self.buffer.len() < total {
                break;
            }
            let body = &self.buffer[4..total];
            let text = std::str::from_utf8(body)?;
            messages.push(serde_json::from_str(text)?);
            self.buffer.drain(..total);
        }

        Ok(messages)
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

pub fn decode_frames(chunk: &[u8], decoder: &mut FrameDecoder) -> Result<Vec<serde_json::Value>, FramingError> {
    decoder.push(chunk)
}
