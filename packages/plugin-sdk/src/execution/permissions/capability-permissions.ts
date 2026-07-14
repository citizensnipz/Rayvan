import type { PluginCapabilityPermissionPolicy } from "./types.js";

/**
 * Default policy: no capability requires permissions.
 * Inject a non-empty policy when testing or enforcing host grants.
 */
export const DEFAULT_CAPABILITY_PERMISSIONS: PluginCapabilityPermissionPolicy =
  {
    authenticate: [],
    discover: [],
    inspect: [],
    plan: [],
    apply: [],
    verify: [],
  };
