import type { PluginCapability } from "../manifest/index.js";

/** Default per-capability timeouts in milliseconds. */
export const DEFAULT_PLUGIN_TIMEOUTS: Readonly<
  Record<PluginCapability, number>
> = {
  authenticate: 120_000,
  discover: 60_000,
  inspect: 30_000,
  plan: 30_000,
  apply: 120_000,
  verify: 60_000,
  evaluate_findings: 30_000,
};
