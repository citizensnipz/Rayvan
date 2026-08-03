/**
 * Plugin IPC framing: u32 big-endian payload length + UTF-8 JSON bytes.
 * (Daemon IPC uses little-endian; plugin protocol is deliberately BE.)
 */

export function encodePluginFrame(payload: unknown): Buffer {
  const body = Buffer.from(JSON.stringify(payload), "utf8");
  const header = Buffer.alloc(4);
  header.writeUInt32BE(body.length, 0);
  return Buffer.concat([header, body]);
}

export class PluginFrameDecoder {
  private buffer = Buffer.alloc(0);

  push(chunk: Buffer): unknown[] {
    this.buffer = Buffer.concat([this.buffer, chunk]);
    const messages: unknown[] = [];

    while (this.buffer.length >= 4) {
      const length = this.buffer.readUInt32BE(0);
      if (length > 16 * 1024 * 1024) {
        throw new Error(`Plugin frame length ${length} exceeds maximum`);
      }
      if (this.buffer.length < 4 + length) {
        break;
      }
      const body = this.buffer.subarray(4, 4 + length);
      this.buffer = this.buffer.subarray(4 + length);
      messages.push(JSON.parse(body.toString("utf8")));
    }

    return messages;
  }

  reset(): void {
    this.buffer = Buffer.alloc(0);
  }
}
