import type { RayvanPlugin } from "../plugin.js";
import { InProcessPluginRegistry } from "../registry/index.js";
import { NoopPluginExecutionEventSink } from "./events/noop.js";
import type { PluginExecutionEventSink } from "./events/types.js";
import { DEFAULT_CAPABILITY_PERMISSIONS } from "./permissions/capability-permissions.js";
import { InMemoryPluginPermissionResolver } from "./permissions/memory-resolver.js";
import type {
  PluginCapabilityPermissionPolicy,
  PluginPermissionResolver,
} from "./permissions/types.js";
import { InProcessPluginRuntime } from "./runtime/in-process.js";
import type { PluginRuntime } from "./runtime/types.js";
import {
  PluginExecutionService,
  type PluginExecutionServiceOptions,
} from "./service.js";

export interface CreatePluginExecutionStackOptions {
  plugins?: RayvanPlugin[];
  permissionResolver?: PluginPermissionResolver;
  eventSink?: PluginExecutionEventSink;
  capabilityPermissions?: PluginCapabilityPermissionPolicy;
  idFactory?: () => string;
  now?: () => Date;
  runtime?: PluginRuntime;
}

export interface PluginExecutionStack {
  registry: InProcessPluginRegistry;
  runtime: PluginRuntime;
  permissionResolver: PluginPermissionResolver;
  eventSink: PluginExecutionEventSink;
  executionService: PluginExecutionService;
}

/**
 * Convenience factory that wires registry, in-process runtime, permissions,
 * events, and PluginExecutionService for local/dev/test use.
 */
export function createPluginExecutionStack(
  options: CreatePluginExecutionStackOptions = {},
): PluginExecutionStack {
  const registry = new InProcessPluginRegistry();
  for (const plugin of options.plugins ?? []) {
    registry.register(plugin);
  }

  const runtime = options.runtime ?? new InProcessPluginRuntime(registry);
  const permissionResolver =
    options.permissionResolver ?? new InMemoryPluginPermissionResolver();
  const eventSink = options.eventSink ?? new NoopPluginExecutionEventSink();

  const serviceOptions: PluginExecutionServiceOptions = {
    registry,
    runtime,
    permissionResolver,
    eventSink,
    capabilityPermissions:
      options.capabilityPermissions ?? DEFAULT_CAPABILITY_PERMISSIONS,
    idFactory: options.idFactory,
    now: options.now,
  };

  return {
    registry,
    runtime,
    permissionResolver,
    eventSink,
    executionService: new PluginExecutionService(serviceOptions),
  };
}
