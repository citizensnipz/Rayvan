import { describe, expect, it } from "vitest";

import { encodeFrame, FrameDecoder } from "../src/framing.js";

describe("frame codec", () => {
  it("round-trips a JSON message", () => {
    const decoder = new FrameDecoder();
    const payload = { jsonrpc: "2.0", id: "1", method: "system.ping" };
    const messages = decoder.push(encodeFrame(payload));
    expect(messages).toEqual([payload]);
  });

  it("handles split chunks", () => {
    const decoder = new FrameDecoder();
    const frame = encodeFrame({ ok: true });
    const first = frame.subarray(0, 3);
    const second = frame.subarray(3);
    expect(decoder.push(first)).toEqual([]);
    expect(decoder.push(second)).toEqual([{ ok: true }]);
  });
});
