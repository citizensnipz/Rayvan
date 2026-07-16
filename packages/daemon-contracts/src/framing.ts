/**
 * Length-prefixed UTF-8 JSON frames over stream transports.
 * Format: u32 little-endian payload length + UTF-8 JSON bytes.
 */

export function encodeFrame(payload: unknown): Buffer {
  const body = Buffer.from(JSON.stringify(payload), "utf8");
  const header = Buffer.alloc(4);
  header.writeUInt32LE(body.length, 0);
  return Buffer.concat([header, body]);
}

export class FrameDecoder {
  private buffer = Buffer.alloc(0);

  push(chunk: Buffer): unknown[] {
    this.buffer = Buffer.concat([this.buffer, chunk]);
    const messages: unknown[] = [];

    while (this.buffer.length >= 4) {
      const length = this.buffer.readUInt32LE(0);
      if (length > 16 * 1024 * 1024) {
        throw new Error(`Frame length ${length} exceeds maximum`);
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
