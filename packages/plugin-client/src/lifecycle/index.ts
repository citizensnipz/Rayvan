export type PluginLifecycleState =
  | "stopped"
  | "starting"
  | "running"
  | "stopping"
  | "crashed";

export interface PluginLifecycleStatus {
  pluginId: string;
  state: PluginLifecycleState;
  lastError?: string;
}
