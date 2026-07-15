import type { PluginCapability, PluginPermission } from "../../manifest/index.js";
import type { PluginExecutionActor } from "../actor.js";

export type PluginCapabilityPermissionPolicy = Record<
  PluginCapability,
  readonly PluginPermission[]
>;

export interface PluginPermissionResolveContext {
  pluginId: string;
  capability: PluginCapability;
  actor: PluginExecutionActor;
  projectId?: string;
  environmentId?: string;
  resourceId?: string;
}

export interface PluginPermissionResolver {
  resolve(
    context: PluginPermissionResolveContext,
  ): Promise<readonly PluginPermission[]> | readonly PluginPermission[];
}
