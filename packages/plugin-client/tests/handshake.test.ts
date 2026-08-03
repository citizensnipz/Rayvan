import { spawn } from "node:child_process";
import { writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { OutOfProcessPluginRuntime } from "../src/runtime/out-of-process.js";

const mockWorkerSource = `
import { encodePluginFrame, PluginFrameDecoder } from ${JSON.stringify(
  fileURLToPath(new URL("../src/framing.ts", import.meta.url)),
)};

const decoder = new PluginFrameDecoder();
process.stdin.on("data", (chunk) => {
  for (const message of decoder.push(chunk)) {
    const req = message;
    if (req.method === "initialize") {
      process.stdout.write(encodePluginFrame({
        jsonrpc: "2.0",
        id: req.id,
        result: {
          protocolVersion: "1",
          pluginId: "io.rayvan.github",
          capabilities: ["discover"],
        },
      }));
    } else if (req.method === "discover") {
      process.stdout.write(encodePluginFrame({
        jsonrpc: "2.0",
        id: req.id,
        result: [{
          providerResourceId: "acme/demo",
          resourceType: "github.repository",
          name: "acme/demo",
          metadata: {},
          schemaVersion: "1.0.0",
        }],
      }));
    } else if (req.method === "shutdown") {
      process.stdout.write(encodePluginFrame({
        jsonrpc: "2.0",
        id: req.id,
        result: { ok: true },
      }));
      process.exit(0);
    } else {
      process.stdout.write(encodePluginFrame({
        jsonrpc: "2.0",
        id: req.id,
        error: { code: -32601, message: "Method not found" },
      }));
    }
  }
});
`;

describe("OutOfProcessPluginRuntime handshake", () => {
  it("initializes and invokes discover over framed JSON-RPC", async () => {
    const dir = mkdtempSync(join(tmpdir(), "rayvan-oop-"));
    // Use a self-contained JS worker that inlines framing to avoid TS loader issues.
    const workerPath = join(dir, "worker.mjs");
    writeFileSync(
      workerPath,
      `
function encodePluginFrame(payload) {
  const body = Buffer.from(JSON.stringify(payload), "utf8");
  const header = Buffer.alloc(4);
  header.writeUInt32BE(body.length, 0);
  return Buffer.concat([header, body]);
}
class PluginFrameDecoder {
  constructor() { this.buffer = Buffer.alloc(0); }
  push(chunk) {
    this.buffer = Buffer.concat([this.buffer, chunk]);
    const messages = [];
    while (this.buffer.length >= 4) {
      const length = this.buffer.readUInt32BE(0);
      if (this.buffer.length < 4 + length) break;
      const body = this.buffer.subarray(4, 4 + length);
      this.buffer = this.buffer.subarray(4 + length);
      messages.push(JSON.parse(body.toString("utf8")));
    }
    return messages;
  }
}
const decoder = new PluginFrameDecoder();
process.stdin.on("data", (chunk) => {
  for (const req of decoder.push(chunk)) {
    if (req.method === "initialize") {
      process.stdout.write(encodePluginFrame({
        jsonrpc: "2.0", id: req.id,
        result: { protocolVersion: "1", pluginId: "io.rayvan.github", capabilities: ["discover"] },
      }));
    } else if (req.method === "discover") {
      process.stdout.write(encodePluginFrame({
        jsonrpc: "2.0", id: req.id,
        result: [{
          providerResourceId: "acme/demo",
          resourceType: "github.repository",
          name: "acme/demo",
          metadata: {},
          schemaVersion: "1.0.0",
        }],
      }));
    } else if (req.method === "shutdown") {
      process.stdout.write(encodePluginFrame({ jsonrpc: "2.0", id: req.id, result: { ok: true } }));
      process.exit(0);
    }
  }
});
`,
    );

    const runtime = new OutOfProcessPluginRuntime({
      pluginId: "io.rayvan.github",
      executable: process.execPath,
      args: [workerPath],
      spawnFn: (executable, args, opts) =>
        spawn(executable, args, {
          stdio: ["pipe", "pipe", "pipe"],
          env: opts.env,
          cwd: opts.cwd,
        }) as ReturnType<typeof spawn> & {
          stdin: NodeJS.WritableStream;
          stdout: NodeJS.ReadableStream;
          stderr: NodeJS.ReadableStream;
        },
    });

    const result = await runtime.invoke({
      pluginId: "io.rayvan.github",
      capability: "discover",
      input: { pluginId: "io.rayvan.github", integrationId: "c1" },
      signal: AbortSignal.timeout(10_000),
    });

    expect(result).toEqual([
      {
        providerResourceId: "acme/demo",
        resourceType: "github.repository",
        name: "acme/demo",
        metadata: {},
        schemaVersion: "1.0.0",
      },
    ]);

    await runtime.stop();
    void mockWorkerSource;
  });
});
