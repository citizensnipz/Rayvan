import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";

import {
  DEFAULT_PLUGIN_TIMEOUTS,
  PluginCapabilityError,
  PluginExecutionError,
  type PluginRuntime,
  type PluginRuntimeInvocation,
} from "@rayvan/plugin-sdk";

import { PluginHostError, PluginTransportError } from "../errors.js";
import { capabilityToRpcMethod } from "../protocol.js";
import { PluginProcessSession } from "../session.js";

export interface OutOfProcessPluginRuntimeOptions {
  pluginId: string;
  executable: string;
  args?: string[];
  env?: NodeJS.ProcessEnv;
  cwd?: string;
  /** Fallback when a capability has no DEFAULT_PLUGIN_TIMEOUTS entry. */
  requestTimeoutMs?: number;
  /** Optional override for tests (inject a pre-spawned child). */
  spawnFn?: (
    executable: string,
    args: string[],
    options: { env?: NodeJS.ProcessEnv; cwd?: string },
  ) => ChildProcessWithoutNullStreams;
}

/**
 * {@link PluginRuntime} that spawns a native plugin launcher and speaks
 * framed JSON-RPC 2.0 over stdin/stdout. Abort/cancel kills the process.
 */
export class OutOfProcessPluginRuntime implements PluginRuntime {
  private session: PluginProcessSession | null = null;
  private starting: Promise<PluginProcessSession> | null = null;

  constructor(private readonly options: OutOfProcessPluginRuntimeOptions) {}

  async invoke<TInput, TOutput>(
    invocation: PluginRuntimeInvocation<TInput>,
  ): Promise<TOutput> {
    if (invocation.pluginId !== this.options.pluginId) {
      throw new PluginCapabilityError(
        invocation.pluginId,
        invocation.capability,
        `OutOfProcessPluginRuntime is bound to "${this.options.pluginId}"`,
      );
    }

    if (invocation.signal.aborted) {
      throw new PluginExecutionError(
        invocation.pluginId,
        invocation.capability,
        "Plugin invocation aborted before start",
        { cause: invocation.signal.reason },
      );
    }

    const session = await this.ensureSession();
    const method = capabilityToRpcMethod(invocation.capability);
    const timeoutMs =
      this.options.requestTimeoutMs ??
      DEFAULT_PLUGIN_TIMEOUTS[invocation.capability] ??
      30_000;

    const onAbort = () => {
      session.kill("SIGKILL");
      this.session = null;
      this.starting = null;
    };
    invocation.signal.addEventListener("abort", onAbort, { once: true });

    try {
      return await session.request<TOutput>(method, invocation.input, {
        timeoutMs,
      });
    } catch (error) {
      if (invocation.signal.aborted) {
        throw new PluginExecutionError(
          invocation.pluginId,
          invocation.capability,
          "Plugin invocation cancelled",
          { cause: invocation.signal.reason },
        );
      }
      throw error instanceof PluginTransportError
        ? new PluginExecutionError(
            invocation.pluginId,
            invocation.capability,
            error.message,
            { cause: error },
          )
        : error;
    } finally {
      invocation.signal.removeEventListener("abort", onAbort);
    }
  }

  async stop(): Promise<void> {
    if (this.session) {
      await this.session.shutdown();
      this.session = null;
    }
    this.starting = null;
  }

  private async ensureSession(): Promise<PluginProcessSession> {
    if (this.session) return this.session;
    if (this.starting) return this.starting;

    this.starting = (async () => {
      const spawnFn =
        this.options.spawnFn ??
        ((executable, args, opts) =>
          spawn(executable, args, {
            stdio: ["pipe", "pipe", "pipe"],
            env: opts.env,
            cwd: opts.cwd,
            windowsHide: true,
          }) as ChildProcessWithoutNullStreams);

      let child: ChildProcessWithoutNullStreams;
      try {
        child = spawnFn(this.options.executable, this.options.args ?? [], {
          env: this.options.env ?? process.env,
          cwd: this.options.cwd,
        });
      } catch (error) {
        throw new PluginHostError(
          `Failed to start plugin "${this.options.pluginId}": ${
            error instanceof Error ? error.message : String(error)
          }`,
        );
      }

      const session = new PluginProcessSession({
        pluginId: this.options.pluginId,
        child,
        requestTimeoutMs: this.options.requestTimeoutMs,
        onTimeoutKill: () => {
          this.session = null;
          this.starting = null;
        },
      });
      await session.initialize();
      this.session = session;
      return session;
    })();

    try {
      return await this.starting;
    } finally {
      this.starting = null;
    }
  }
}
