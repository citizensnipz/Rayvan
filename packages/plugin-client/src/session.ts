import { randomUUID } from "node:crypto";
import type { ChildProcessWithoutNullStreams } from "node:child_process";

import { encodePluginFrame, PluginFrameDecoder } from "./framing.js";
import {
  isJsonRpcResponse,
  PLUGIN_PROTOCOL_VERSION,
  type JsonRpcRequest,
  type JsonRpcResponse,
  type PluginInitializeResult,
  type PluginRpcMethod,
} from "./protocol.js";
import { PluginTransportError } from "./errors.js";
import { redactPluginLogText } from "./redact.js";

export interface PluginProcessSessionOptions {
  pluginId: string;
  child: ChildProcessWithoutNullStreams;
  requestTimeoutMs?: number;
  /** Called when the process is killed due to RPC timeout. */
  onTimeoutKill?: () => void;
}

type Pending = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
};

export interface PluginRequestOptions {
  timeoutMs?: number;
}

/**
 * Host-side framed JSON-RPC session over a child process stdin/stdout.
 * stderr is logs only and is never parsed as RPC.
 */
export class PluginProcessSession {
  readonly pluginId: string;
  private readonly child: ChildProcessWithoutNullStreams;
  private readonly decoder = new PluginFrameDecoder();
  private readonly pending = new Map<string | number, Pending>();
  private readonly requestTimeoutMs: number;
  private readonly onTimeoutKill?: () => void;
  private closed = false;
  private initializePromise: Promise<PluginInitializeResult> | null = null;

  constructor(options: PluginProcessSessionOptions) {
    this.pluginId = options.pluginId;
    this.child = options.child;
    this.requestTimeoutMs = options.requestTimeoutMs ?? 30_000;
    this.onTimeoutKill = options.onTimeoutKill;

    this.child.stdout.on("data", (chunk: Buffer) => {
      try {
        for (const message of this.decoder.push(chunk)) {
          this.handleMessage(message);
        }
      } catch (error) {
        this.failAll(
          error instanceof Error ? error : new Error(String(error)),
        );
      }
    });

    this.child.stderr.on("data", (chunk: Buffer) => {
      const text = redactPluginLogText(chunk.toString("utf8").trimEnd());
      if (text.length > 0) {
        console.error(`[plugin:${this.pluginId}] ${text}`);
      }
    });

    this.child.on("exit", (code, signal) => {
      this.closed = true;
      this.failAll(
        new PluginTransportError(
          `Plugin process exited (code=${code}, signal=${signal})`,
        ),
      );
    });
  }

  get pid(): number | undefined {
    return this.child.pid;
  }

  async initialize(): Promise<PluginInitializeResult> {
    if (!this.initializePromise) {
      this.initializePromise = this.request<PluginInitializeResult>(
        "initialize",
        {
          protocolVersion: PLUGIN_PROTOCOL_VERSION,
          pluginId: this.pluginId,
        },
      );
    }
    return this.initializePromise;
  }

  async request<TResult>(
    method: PluginRpcMethod,
    params?: unknown,
    options?: PluginRequestOptions,
  ): Promise<TResult> {
    if (this.closed) {
      throw new PluginTransportError("Plugin process session is closed");
    }
    const id = randomUUID();
    const payload: JsonRpcRequest = {
      jsonrpc: "2.0",
      id,
      method,
      params,
    };
    const timeoutMs = options?.timeoutMs ?? this.requestTimeoutMs;

    return new Promise<TResult>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        this.kill("SIGKILL");
        this.onTimeoutKill?.();
        reject(
          new PluginTransportError(
            `Plugin RPC timeout for method "${method}" after ${timeoutMs}ms`,
          ),
        );
      }, timeoutMs);

      this.pending.set(id, {
        resolve: (value) => resolve(value as TResult),
        reject,
        timer,
      });

      try {
        this.child.stdin.write(encodePluginFrame(payload));
      } catch (error) {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(
          error instanceof Error
            ? error
            : new PluginTransportError(String(error)),
        );
      }
    });
  }

  async shutdown(): Promise<void> {
    try {
      await this.request("shutdown", {});
    } catch {
      // best-effort
    }
    this.kill("SIGTERM");
  }

  kill(signal: NodeJS.Signals = "SIGKILL"): void {
    this.closed = true;
    if (!this.child.killed) {
      this.child.kill(signal);
    }
    this.failAll(new PluginTransportError("Plugin process killed"));
  }

  private handleMessage(message: unknown): void {
    if (!isJsonRpcResponse(message)) {
      return;
    }
    const response = message as JsonRpcResponse;
    if (response.id === null || response.id === undefined) {
      return;
    }
    const pending = this.pending.get(response.id);
    if (!pending) return;
    clearTimeout(pending.timer);
    this.pending.delete(response.id);
    if ("error" in response) {
      pending.reject(
        new PluginTransportError(
          `${response.error.message} (${response.error.code})`,
        ),
      );
      return;
    }
    pending.resolve(response.result);
  }

  private failAll(error: Error): void {
    for (const [, pending] of this.pending) {
      clearTimeout(pending.timer);
      pending.reject(error);
    }
    this.pending.clear();
  }
}
