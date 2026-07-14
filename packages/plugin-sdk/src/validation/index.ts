import type { PluginManifest } from "../manifest/index.js";

export function validatePluginManifest(manifest: PluginManifest): void {
  if (!manifest.id.trim()) {
    throw new Error("Plugin manifest id is required");
  }
  if (!manifest.name.trim()) {
    throw new Error("Plugin manifest name is required");
  }
  if (!manifest.version.trim()) {
    throw new Error("Plugin manifest version is required");
  }
  if (!manifest.protocolVersion.trim()) {
    throw new Error("Plugin manifest protocolVersion is required");
  }
}
