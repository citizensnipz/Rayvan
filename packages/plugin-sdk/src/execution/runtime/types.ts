import type { PluginCapability } from "../../manifest/index.js";

export interface PluginRuntimeInvocation<TInput> {
  pluginId: string;
  capability: PluginCapability;
  input: TInput;
  signal: AbortSignal;
}

/**
 * Trusted invocation boundary for plugin handlers.
 * The execution service must call through this rather than handlers directly.
 */
export interface PluginRuntime {
  invoke<TInput, TOutput>(
    invocation: PluginRuntimeInvocation<TInput>,
  ): Promise<TOutput>;
}
