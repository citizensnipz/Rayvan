import type { RayvanPlugin } from "@rayvan/plugin-sdk";
import { CAPABILITY_HANDLER_KEYS } from "@rayvan/plugin-sdk";

import { encodePluginFrame, PluginFrameDecoder } from "../framing.js";
import {
  PLUGIN_PROTOCOL_VERSION,
  type JsonRpcRequest,
  type JsonRpcResponse,
  type PluginInitializeResult,
  type PluginRpcMethod,
} from "../protocol.js";

/**
 * Serve a {@link RayvanPlugin} over stdin/stdout framed JSON-RPC.
 * Intended for Node SEA / launcher entrypoints. stderr is for logs only.
 */
export async function serveRayvanPlugin(plugin: RayvanPlugin): Promise<void> {
  const decoder = new PluginFrameDecoder();
  let shuttingDown = false;

  const write = (response: JsonRpcResponse) => {
    process.stdout.write(encodePluginFrame(response));
  };

  const handle = async (message: unknown): Promise<void> => {
    const request = message as JsonRpcRequest;
    if (
      !request ||
      request.jsonrpc !== "2.0" ||
      typeof request.method !== "string"
    ) {
      return;
    }

    const id = request.id ?? null;
    try {
      const result = await dispatch(plugin, request.method as PluginRpcMethod, request.params);
      write({ jsonrpc: "2.0", id, result });
      if (request.method === "shutdown") {
        shuttingDown = true;
        process.exit(0);
      }
    } catch (error) {
      write({
        jsonrpc: "2.0",
        id,
        error: {
          code: -32000,
          message:
            error instanceof Error ? error.message : "Plugin handler failed",
        },
      });
    }
  };

  process.stdin.on("data", (chunk: Buffer) => {
    void (async () => {
      for (const message of decoder.push(chunk)) {
        await handle(message);
      }
    })();
  });

  process.stdin.on("end", () => {
    if (!shuttingDown) {
      process.exit(0);
    }
  });
}

async function dispatch(
  plugin: RayvanPlugin,
  method: PluginRpcMethod,
  params: unknown,
): Promise<unknown> {
  switch (method) {
    case "initialize": {
      const result: PluginInitializeResult = {
        protocolVersion: PLUGIN_PROTOCOL_VERSION,
        pluginId: plugin.manifest.id,
        capabilities: plugin.manifest.capabilities,
      };
      return result;
    }
    case "shutdown":
      return { ok: true };
    case "cancel":
      return { ok: true };
    case "authenticate":
    case "discover":
    case "inspect":
    case "plan":
    case "apply":
    case "verify":
    case "evaluate_findings": {
      const capability =
        method === "evaluate_findings" ? "evaluate_findings" : method;
      const handlerKey = CAPABILITY_HANDLER_KEYS[capability];
      const handler = plugin[handlerKey];
      if (typeof handler !== "function") {
        throw new Error(`Capability "${capability}" is not implemented`);
      }
      return (handler as (input: unknown) => Promise<unknown>)(params);
    }
    default:
      throw new Error(`Unknown method "${method}"`);
  }
}
