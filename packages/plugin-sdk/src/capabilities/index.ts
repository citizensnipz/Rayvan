import type { PluginCapability } from "../manifest/index.js";

export function supportsCapability(
  manifest: { capabilities: readonly PluginCapability[] },
  capability: PluginCapability,
): boolean {
  return manifest.capabilities.includes(capability);
}
