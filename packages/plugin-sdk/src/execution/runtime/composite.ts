import type {
  PluginRuntime,
  PluginRuntimeInvocation,
} from "./types.js";

/**
 * Routes invocations to a per-plugin runtime, with an optional fallback
 * (typically {@link InProcessPluginRuntime} for built-ins / tests).
 */
export class CompositePluginRuntime implements PluginRuntime {
  constructor(
    private readonly byPluginId: ReadonlyMap<string, PluginRuntime>,
    private readonly fallback?: PluginRuntime,
  ) {}

  async invoke<TInput, TOutput>(
    invocation: PluginRuntimeInvocation<TInput>,
  ): Promise<TOutput> {
    const runtime =
      this.byPluginId.get(invocation.pluginId) ?? this.fallback;
    if (!runtime) {
      throw new Error(
        `No plugin runtime registered for "${invocation.pluginId}"`,
      );
    }
    return runtime.invoke(invocation);
  }
}
