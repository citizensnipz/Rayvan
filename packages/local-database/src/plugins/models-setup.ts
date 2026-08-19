export type PluginSetupSessionStatus =
  | "active"
  | "completed"
  | "cancelled"
  | "expired";

export type PluginSetupAuthMethod = "github_device_flow" | "pat";

/**
 * Non-secret wizard state for a plugin setup session.
 * Tokens are stored only via CredentialStore.
 */
export interface PluginSetupSessionRecord {
  id: string;
  pluginId: string;
  installedPluginId: string;
  projectId?: string;
  status: PluginSetupSessionStatus;
  currentStepId?: string;
  authMethod?: PluginSetupAuthMethod;
  /** Serializable non-secret state (user codes, selected repos, etc.). */
  state: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
  completedAt?: string;
}
