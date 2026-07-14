import { RAYVAN_PLUGIN_API_VERSION, type PluginManifest } from "@rayvan/plugin-sdk";

/**
 * Serializable catalog of plugin manifests available to the Integrations UI.
 *
 * The desktop webview must not import `plugins/*` packages directly (see
 * `docs/architecture/plugin-system.md`) — real plugin manifests are copied
 * here by hand. Keep the `vercel` / `supabase` / `github` / `example-local`
 * entries in sync with their source manifests when those change.
 *
 * `sentry` and `runpod` are fixture-only catalog entries with no
 * corresponding `plugins/*` package; they exist purely to give the
 * development fixtures and the "Add integration" library richer, more
 * realistic demo data.
 */

export const VERCEL_MANIFEST: PluginManifest = {
  id: "vercel",
  name: "Vercel",
  description: "Connect Rayvan to Vercel projects and deployments.",
  version: "0.0.1",
  publisher: "rayvan",
  rayvanApiVersion: RAYVAN_PLUGIN_API_VERSION,
  capabilities: [],
  permissions: [],
  resourceTypes: [],
  presentation: {
    icon: { iconId: "vercel", initials: "V", label: "Vercel" },
    theme: {
      surface: "dark",
      accentColor: "#FFFFFF",
      foregroundMode: "light",
    },
    supportsMultipleConnections: true,
  },
};

export const SUPABASE_MANIFEST: PluginManifest = {
  id: "supabase",
  name: "Supabase",
  description: "Connect Rayvan to Supabase projects and databases.",
  version: "0.0.1",
  publisher: "rayvan",
  rayvanApiVersion: RAYVAN_PLUGIN_API_VERSION,
  capabilities: [],
  permissions: [],
  resourceTypes: [],
  presentation: {
    icon: { iconId: "supabase", initials: "S", label: "Supabase" },
    theme: {
      surface: "dark",
      accentColor: "#3ECF8E",
      foregroundMode: "light",
    },
    supportsMultipleConnections: true,
  },
};

export const GITHUB_MANIFEST: PluginManifest = {
  id: "github",
  name: "GitHub",
  description: "Connect Rayvan to GitHub repositories and pull requests.",
  version: "0.0.1",
  publisher: "rayvan",
  rayvanApiVersion: RAYVAN_PLUGIN_API_VERSION,
  capabilities: [],
  permissions: [],
  resourceTypes: [],
  presentation: {
    icon: { iconId: "github", initials: "GH", label: "GitHub" },
    theme: {
      surface: "dark",
      accentColor: "#F0F6FC",
      foregroundMode: "light",
    },
    supportsMultipleConnections: true,
  },
};

export const EXAMPLE_LOCAL_MANIFEST: PluginManifest = {
  id: "example-local",
  name: "Example Local",
  description:
    "Built-in mock local environment plugin that demonstrates the Rayvan plugin lifecycle without external credentials.",
  version: "0.1.0",
  publisher: "rayvan",
  rayvanApiVersion: RAYVAN_PLUGIN_API_VERSION,
  capabilities: ["discover", "inspect", "plan", "apply", "verify"],
  permissions: [],
  resourceTypes: [
    {
      id: "local.service",
      name: "Local Service",
      description: "An in-memory mock local development service",
      schemaVersion: "1.0.0",
    },
  ],
  presentation: {
    icon: {
      iconId: "example-local",
      initials: "EL",
      label: "Example Local",
    },
    theme: {
      surface: "neutral",
      accentColor: "#64748B",
      foregroundMode: "dark",
    },
    supportsMultipleConnections: false,
  },
};

/** Fixture-only manifest: no `plugins/sentry` package exists yet. */
export const SENTRY_MANIFEST: PluginManifest = {
  id: "sentry",
  name: "Sentry",
  description: "Connect Rayvan to Sentry projects for error monitoring.",
  version: "0.0.1",
  publisher: "rayvan",
  rayvanApiVersion: RAYVAN_PLUGIN_API_VERSION,
  capabilities: [],
  permissions: [],
  resourceTypes: [],
  presentation: {
    icon: { iconId: "sentry", initials: "SN", label: "Sentry" },
    theme: {
      surface: "brand",
      accentColor: "#362D59",
      foregroundMode: "light",
    },
    supportsMultipleConnections: true,
  },
};

/**
 * Fixture-only manifest for a third-party (non-`rayvan`-published) plugin.
 * Used to demonstrate a catalog entry that is installed but has no
 * connection yet, and that does not carry a "Built-in" / "Official" badge.
 */
export const RUNPOD_MANIFEST: PluginManifest = {
  id: "runpod",
  name: "RunPod",
  description: "Connect Rayvan to RunPod GPU compute instances.",
  version: "0.3.2",
  publisher: "runpod-community",
  rayvanApiVersion: RAYVAN_PLUGIN_API_VERSION,
  capabilities: [],
  permissions: [],
  resourceTypes: [],
  presentation: {
    icon: { iconId: "runpod", initials: "RP", label: "RunPod" },
    theme: {
      surface: "dark",
      accentColor: "#673AB8",
      foregroundMode: "light",
    },
    supportsMultipleConnections: true,
  },
};

export const INTEGRATIONS_CATALOG_MANIFESTS: readonly PluginManifest[] = [
  VERCEL_MANIFEST,
  SUPABASE_MANIFEST,
  GITHUB_MANIFEST,
  SENTRY_MANIFEST,
  EXAMPLE_LOCAL_MANIFEST,
  RUNPOD_MANIFEST,
];
