export const PLUGIN_PROTOCOL_VERSION = "1";

export type PluginPhase =
  | "connect"
  | "discover"
  | "inspect"
  | "plan"
  | "approve"
  | "execute"
  | "audit";

export const PLUGIN_PHASES: readonly PluginPhase[] = [
  "connect",
  "discover",
  "inspect",
  "plan",
  "approve",
  "execute",
  "audit",
] as const;
