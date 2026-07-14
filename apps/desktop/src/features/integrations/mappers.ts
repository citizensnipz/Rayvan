import type {
  InstalledPluginRecord,
  PluginConnectionRecord,
  PluginPermissionGrantRecord,
} from "@rayvan/local-database";
import type { PluginManifest } from "@rayvan/plugin-sdk";

import { formatDateTime } from "../../lib/format.js";
import type {
  IntegrationCardAction,
  IntegrationField,
  IntegrationFieldGroup,
  IntegrationIconViewModel,
  IntegrationStatus,
  IntegrationThemeViewModel,
  LibraryPluginBadge,
  LibraryPluginViewModel,
  PluginIntegrationCardViewModel,
  PluginIntegrationDetailViewModel,
} from "./view-models.js";

const STATUS_LABELS: Record<IntegrationStatus, string> = {
  connected: "Connected",
  syncing: "Syncing",
  attention_required: "Attention required",
  expired: "Expired",
  disconnected: "Disconnected",
  error: "Error",
};

function readMetadataFields(
  metadata: Record<string, unknown>,
  key: "uiCard" | "uiDetail",
): IntegrationField[] {
  const group = metadata[key];
  if (
    typeof group !== "object" ||
    group === null ||
    !Array.isArray((group as { fields?: unknown }).fields)
  ) {
    return [];
  }
  return ((group as { fields: unknown[] }).fields).filter(
    (field): field is IntegrationField =>
      typeof field === "object" &&
      field !== null &&
      typeof (field as IntegrationField).label === "string" &&
      typeof (field as IntegrationField).value === "string",
  );
}

/**
 * Maps a raw `PluginConnectionStatus` to the UI's richer status set,
 * honoring an optional `metadata.uiStatus` override (e.g. a connected
 * integration that needs attention). This is the single place that
 * understands the mapping — cards/detail views never branch on status.
 */
export function resolveIntegrationStatus(
  connection: PluginConnectionRecord,
): { status: IntegrationStatus; label: string } {
  const override = connection.metadata?.uiStatus;
  if (
    typeof override === "string" &&
    Object.prototype.hasOwnProperty.call(STATUS_LABELS, override)
  ) {
    const status = override as IntegrationStatus;
    return { status, label: STATUS_LABELS[status] };
  }

  const status: IntegrationStatus = (() => {
    switch (connection.status) {
      case "connected":
        return "connected";
      case "pending":
        return "syncing";
      case "expired":
        return "expired";
      case "revoked":
      case "disconnected":
        return "disconnected";
      case "error":
        return "error";
      default:
        return "disconnected";
    }
  })();

  return { status, label: STATUS_LABELS[status] };
}

export function mapIconFromManifest(manifest: PluginManifest): IntegrationIconViewModel {
  const icon = manifest.presentation?.icon;
  const label = icon?.label ?? manifest.name;
  const initials = icon?.initials ?? manifest.name.slice(0, 2).toUpperCase();
  return { iconId: icon?.iconId, initials, label };
}

export function mapThemeFromManifest(manifest: PluginManifest): IntegrationThemeViewModel {
  const theme = manifest.presentation?.theme;
  return {
    surface: theme?.surface ?? "neutral",
    accentColor: theme?.accentColor,
    foregroundMode: theme?.foregroundMode,
  };
}

function cardActionsFor(): IntegrationCardAction[] {
  return [
    { id: "open", label: "Open", kind: "primary" },
    { id: "sync", label: "Sync", kind: "secondary" },
    { id: "configure", label: "Configure", kind: "secondary" },
  ];
}

export function mapConnectionToCardViewModel(
  connection: PluginConnectionRecord,
  installed: InstalledPluginRecord,
): PluginIntegrationCardViewModel {
  const manifest = installed.manifestSnapshot;
  const { status, label } = resolveIntegrationStatus(connection);

  return {
    connectionId: connection.id,
    installedPluginId: installed.id,
    pluginId: manifest.id,
    connectionName: connection.name,
    pluginName: manifest.name,
    publisher: manifest.publisher,
    description: manifest.description,
    icon: mapIconFromManifest(manifest),
    theme: mapThemeFromManifest(manifest),
    status,
    statusLabel: label,
    fields: readMetadataFields(connection.metadata, "uiCard"),
    actions: cardActionsFor(),
  };
}

export function mapConnectionToDetailViewModel(
  connection: PluginConnectionRecord,
  installed: InstalledPluginRecord,
  grants: PluginPermissionGrantRecord[],
): PluginIntegrationDetailViewModel {
  const manifest = installed.manifestSnapshot;
  const { status, label } = resolveIntegrationStatus(connection);

  const connectionGroup: IntegrationFieldGroup = {
    title: "Connection",
    fields: [
      { label: "Status", value: label },
      {
        label: "Last sync",
        value: connection.lastSuccessfulSyncAt
          ? formatDateTime(connection.lastSuccessfulSyncAt)
          : "Never",
      },
      {
        label: "Connected since",
        value: connection.lastAuthenticatedAt
          ? formatDateTime(connection.lastAuthenticatedAt)
          : formatDateTime(connection.createdAt),
      },
      ...(connection.externalAccountName
        ? [{ label: "Account", value: connection.externalAccountName }]
        : []),
    ],
  };

  const detailFields = readMetadataFields(connection.metadata, "uiDetail");
  const detailsGroup: IntegrationFieldGroup | null =
    detailFields.length > 0
      ? { title: `${manifest.name} details`, fields: detailFields }
      : null;

  const activeGrants = grants.filter((grant) => grant.granted && !grant.revokedAt);
  const permissionsGroup: IntegrationFieldGroup = {
    title: "Permissions",
    fields:
      activeGrants.length > 0
        ? activeGrants.map((grant) => ({
            label: grant.permission,
            value: "Granted",
          }))
        : [{ label: "Permissions", value: "None granted" }],
  };

  return {
    connectionId: connection.id,
    installedPluginId: installed.id,
    pluginId: manifest.id,
    connectionName: connection.name,
    pluginName: manifest.name,
    publisher: manifest.publisher,
    version: manifest.version,
    description: manifest.description,
    icon: mapIconFromManifest(manifest),
    theme: mapThemeFromManifest(manifest),
    status,
    statusLabel: label,
    overview: {
      groups: [connectionGroup, ...(detailsGroup ? [detailsGroup] : []), permissionsGroup],
    },
    resources: {
      groups: [],
      isEmpty: true,
    },
    configuration: {
      grants: grants.map((grant) => ({
        permission: grant.permission,
        granted: grant.granted && !grant.revokedAt,
        grantedAt: grant.grantedAt,
      })),
      actions: [],
    },
    activity: {
      isEmpty: true,
      items: [],
    },
  };
}

function badgeFor(installed: InstalledPluginRecord): LibraryPluginBadge {
  if (installed.source.type === "built_in") {
    return "built-in";
  }
  if (installed.manifestSnapshot.publisher === "rayvan") {
    return "official";
  }
  return null;
}

export function mapInstalledPluginToLibraryViewModel(
  installed: InstalledPluginRecord,
  connectionsForProject: PluginConnectionRecord[],
): LibraryPluginViewModel {
  const manifest = installed.manifestSnapshot;
  const supportsMultipleConnections =
    manifest.presentation?.supportsMultipleConnections ?? false;
  const existingConnectionCount = connectionsForProject.filter(
    (connection) =>
      connection.installedPluginId === installed.id &&
      connection.status !== "disconnected" &&
      connection.status !== "revoked",
  ).length;

  return {
    installedPluginId: installed.id,
    pluginId: manifest.id,
    name: manifest.name,
    publisher: manifest.publisher,
    description: manifest.description,
    version: manifest.version,
    icon: mapIconFromManifest(manifest),
    theme: mapThemeFromManifest(manifest),
    badge: badgeFor(installed),
    supportsMultipleConnections,
    existingConnectionCount,
    eligible: existingConnectionCount === 0 || supportsMultipleConnections,
  };
}
