import type {
  PluginForegroundMode,
  PluginThemeSurface,
} from "@rayvan/plugin-sdk";

/**
 * Serializable, desktop-owned view-models for the Integrations UI.
 * Card/detail data derives from `PluginConnectionRecord` +
 * `InstalledPluginRecord.manifestSnapshot.presentation` — never Core
 * `Integration`. Actions carry string ids, never functions, so these
 * view-models stay serializable.
 */

export type IntegrationStatus =
  | "connected"
  | "syncing"
  | "attention_required"
  | "expired"
  | "disconnected"
  | "error";

export interface IntegrationField {
  label: string;
  value: string;
}

export interface IntegrationIconViewModel {
  iconId?: string;
  initials?: string;
  label: string;
}

export interface IntegrationThemeViewModel {
  surface: PluginThemeSurface;
  accentColor?: string;
  foregroundMode?: PluginForegroundMode;
}

export type IntegrationCardActionId = "open" | "sync" | "configure";

export interface IntegrationCardAction {
  id: IntegrationCardActionId;
  label: string;
  kind: "primary" | "secondary";
}

export interface PluginIntegrationCardViewModel {
  connectionId: string;
  installedPluginId: string;
  pluginId: string;
  /** The user-chosen connection name, e.g. "Acme Organisation". */
  connectionName: string;
  /** The plugin's display name, e.g. "Vercel". */
  pluginName: string;
  publisher: string;
  description?: string;
  icon: IntegrationIconViewModel;
  theme: IntegrationThemeViewModel;
  status: IntegrationStatus;
  statusLabel: string;
  fields: IntegrationField[];
  actions: IntegrationCardAction[];
}

export interface IntegrationFieldGroup {
  title: string;
  fields: IntegrationField[];
}

export interface PluginIntegrationDetailViewModel {
  connectionId: string;
  installedPluginId: string;
  pluginId: string;
  connectionName: string;
  pluginName: string;
  publisher: string;
  version: string;
  description?: string;
  icon: IntegrationIconViewModel;
  theme: IntegrationThemeViewModel;
  status: IntegrationStatus;
  statusLabel: string;
  overview: {
    groups: IntegrationFieldGroup[];
  };
  resources: {
    groups: IntegrationFieldGroup[];
    isEmpty: boolean;
  };
  configuration: {
    grants: Array<{ permission: string; granted: boolean; grantedAt: string }>;
    actions: IntegrationCardAction[];
  };
  activity: {
    isEmpty: boolean;
    items: Array<{ label: string; timestamp?: string }>;
  };
}

export type LibraryPluginBadge = "built-in" | "official" | null;

export interface LibraryPluginViewModel {
  installedPluginId: string;
  pluginId: string;
  name: string;
  publisher: string;
  description?: string;
  version: string;
  icon: IntegrationIconViewModel;
  theme: IntegrationThemeViewModel;
  badge: LibraryPluginBadge;
  supportsMultipleConnections: boolean;
  existingConnectionCount: number;
  /** Eligible when not yet configured for this project, or when multi-connection is supported. */
  eligible: boolean;
}

export type IntegrationTab =
  | { kind: "home" }
  | { kind: "detail"; connectionId: string; label: string };
