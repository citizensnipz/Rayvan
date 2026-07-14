import { CAPABILITY_HANDLER_KEYS } from "../../capabilities/index.js";
import {
  PluginCapabilityError,
  PluginExecutionError,
  PluginNotFoundError,
} from "../../errors/index.js";
import type { PluginRegistry } from "../../registry/index.js";
import type {
  PluginRuntime,
  PluginRuntimeInvocation,
} from "./types.js";

/**
 * Resolves a validated handler from the registry and invokes it in-process.
 */
export class InProcessPluginRuntime implements PluginRuntime {
  constructor(private readonly registry: PluginRegistry) {}

  async invoke<TInput, TOutput>(
    invocation: PluginRuntimeInvocation<TInput>,
  ): Promise<TOutput> {
    const plugin = this.registry.get(invocation.pluginId);
    if (!plugin) {
      throw new PluginNotFoundError(invocation.pluginId);
    }

    const handlerKey = CAPABILITY_HANDLER_KEYS[invocation.capability];
    const handler = plugin[handlerKey];
    if (typeof handler !== "function") {
      throw new PluginCapabilityError(
        invocation.pluginId,
        invocation.capability,
        `Plugin "${invocation.pluginId}" is missing handler for capability "${invocation.capability}"`,
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

    return (handler as unknown as (input: TInput) => Promise<TOutput>)(
      invocation.input,
    );
  }
}
