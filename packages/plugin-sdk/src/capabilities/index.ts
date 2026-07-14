import type { PluginCapability, PluginManifest } from "../manifest/index.js";

export function supportsCapability(
  manifest: Pick<PluginManifest, "capabilities">,
  capability: PluginCapability,
): boolean {
  return manifest.capabilities.includes(capability);
}

/** Maps each capability to the optional handler property on RayvanPlugin. */
export const CAPABILITY_HANDLER_KEYS = {
  authenticate: "authenticate",
  discover: "discover",
  inspect: "inspect",
  plan: "plan",
  apply: "apply",
  verify: "verify",
} as const satisfies Record<
  PluginCapability,
  Exclude<PluginCapability, never>
>;
