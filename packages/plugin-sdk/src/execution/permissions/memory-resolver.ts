import type { PluginPermission } from "../../manifest/index.js";
import type {
  PluginPermissionResolveContext,
  PluginPermissionResolver,
} from "./types.js";

/**
 * In-memory permission resolver. Returns grants for a plugin id,
 * or an empty list when none are configured.
 */
export class InMemoryPluginPermissionResolver
  implements PluginPermissionResolver
{
  private readonly grantsByPluginId: Map<string, readonly PluginPermission[]>;

  constructor(
    grants: Readonly<Record<string, readonly PluginPermission[]>> = {},
  ) {
    this.grantsByPluginId = new Map(Object.entries(grants));
  }

  setGrants(pluginId: string, grants: readonly PluginPermission[]): void {
    this.grantsByPluginId.set(pluginId, grants);
  }

  resolve(
    context: PluginPermissionResolveContext,
  ): readonly PluginPermission[] {
    return this.grantsByPluginId.get(context.pluginId) ?? [];
  }
}

/** Resolver that grants every known permission (useful for trusted built-ins). */
export class AllowAllPluginPermissionResolver
  implements PluginPermissionResolver
{
  constructor(private readonly permissions: readonly PluginPermission[]) {}

  resolve(
    context: PluginPermissionResolveContext,
  ): readonly PluginPermission[] {
    void context;
    return this.permissions;
  }
}
