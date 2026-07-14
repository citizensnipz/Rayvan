/**
 * End-to-end control-plane lifecycle.
 * Approve and audit are owned by Rayvan Core / host, not by plugins.
 */
export type PluginLifecyclePhase =
  | "authenticate"
  | "discover"
  | "inspect"
  | "plan"
  | "approve"
  | "apply"
  | "verify"
  | "audit";

export const PLUGIN_LIFECYCLE_PHASES: readonly PluginLifecyclePhase[] = [
  "authenticate",
  "discover",
  "inspect",
  "plan",
  "approve",
  "apply",
  "verify",
  "audit",
] as const;
