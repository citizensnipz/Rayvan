import { describe, expect, it } from "vitest";

import { encodePluginFrame, PluginFrameDecoder } from "../src/framing.js";

describe("plugin framing (BE)", () => {
  it("round-trips length-prefixed JSON", () => {
    const frame = encodePluginFrame({ jsonrpc: "2.0", id: 1, result: { ok: true } });
    expect(frame.readUInt32BE(0)).toBe(frame.length - 4);

    const decoder = new PluginFrameDecoder();
    const messages = decoder.push(frame);
    expect(messages).toEqual([{ jsonrpc: "2.0", id: 1, result: { ok: true } }]);
  });

  it("handles split chunks", () => {
    const frame = encodePluginFrame({ hello: "world" });
    const decoder = new PluginFrameDecoder();
    expect(decoder.push(frame.subarray(0, 2))).toEqual([]);
    expect(decoder.push(frame.subarray(2))).toEqual([{ hello: "world" }]);
  });
});
