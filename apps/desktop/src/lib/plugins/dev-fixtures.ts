import type { PluginExecutionActor } from "@rayvan/plugin-sdk";
import type {
  InstalledPluginRecord,
  InstalledPluginRepository,
  PluginConnectionRepository,
  PluginConnectionStatus,
  PluginPermissionGrantRepository,
} from "@rayvan/local-database";

import type { IntegrationStatus } from "../../features/integrations/view-models.js";

import { INTEGRATIONS_CATALOG_MANIFESTS, RUNPOD_MANIFEST } from "./catalog.js";

/**
 * DEVELOPMENT ONLY seed data for the Integrations UI.
 *
 * Populates the in-memory plugin persistence with a realistic catalog of
 * installed plugins and, per project, a handful of pre-connected
 * integrations so the UI has something to render without any real
 * provider credentials or network access. Never used outside the dev
 * fixture gateway (`dev-gateway.ts`).
 */

export const DEV_FIXTURE_SYSTEM_ACTOR: PluginExecutionActor = {
  type: "system",
  id: "dev-fixture-seed",
};

export const DEV_FIXTURE_USER_ACTOR: PluginExecutionActor = {
  type: "user",
  id: "dev-local-user",
  displayName: "You",
};

/** Card/detail field pairs stored under `connection.metadata.uiCard` / `uiDetail`. */
export interface DevUiField {
  label: string;
  value: string;
}

/**
 * Installs every catalog manifest as a built-in (or, for `runpod`,
 * community-sourced) `InstalledPluginRecord` if it is not already present.
 * Idempotent: safe to call multiple times against the same repository.
 */
export async function ensureCatalogInstalled(
  installedPlugins: InstalledPluginRepository,
): Promise<Map<string, InstalledPluginRecord>> {
  const now = new Date().toISOString();
  const byPluginId = new Map<string, InstalledPluginRecord>();

  for (const manifest of INTEGRATIONS_CATALOG_MANIFESTS) {
    const existing = await installedPlugins.getByPluginId(manifest.id);
    if (existing) {
      byPluginId.set(manifest.id, existing);
      continue;
    }

    const isThirdParty = manifest.id === RUNPOD_MANIFEST.id;
    const record: InstalledPluginRecord = {
      id: crypto.randomUUID(),
      pluginId: manifest.id,
      pluginVersion: manifest.version,
      manifestVersion: manifest.version,
      rayvanApiVersion: manifest.rayvanApiVersion,
      source: isThirdParty
        ? { type: "package", packageId: `${manifest.id}-community-plugin` }
        : { type: "built_in" },
      status: "installed",
      enabled: true,
      installedAt: now,
      updatedAt: now,
      lastLoadedAt: now,
      manifestSnapshot: structuredClone(manifest),
    };
    await installedPlugins.save(record);
    byPluginId.set(manifest.id, record);
  }

  return byPluginId;
}

interface SeedConnectionSpec {
  pluginId: string;
  connectionName: string;
  externalAccountName: string;
  status: PluginConnectionStatus;
  /** UI-only status override rendered instead of the raw connection status. */
  uiStatus?: IntegrationStatus;
  fields: DevUiField[];
  grantPermissions: boolean;
}

const SEED_CONNECTIONS: readonly SeedConnectionSpec[] = [
  {
    pluginId: "vercel",
    connectionName: "Acme Organisation",
    externalAccountName: "Acme Organisation",
    status: "connected",
    fields: [
      { label: "Project", value: "rayvan-web" },
      { label: "Environment", value: "Production" },
      { label: "Last deployment", value: "Ready" },
    ],
    grantPermissions: true,
  },
  {
    pluginId: "supabase",
    connectionName: "Production",
    externalAccountName: "Production",
    status: "connected",
    fields: [
      { label: "Project", value: "rayvan-prod" },
      { label: "Region", value: "Singapore" },
      { label: "Database", value: "Healthy" },
    ],
    grantPermissions: true,
  },
  {
    pluginId: "github",
    connectionName: "Rayvan",
    externalAccountName: "Rayvan",
    status: "connected",
    fields: [
      { label: "Repository", value: "rayvan" },
      { label: "Branch", value: "main" },
      { label: "Open pull requests", value: "3" },
    ],
    grantPermissions: true,
  },
  {
    pluginId: "sentry",
    connectionName: "Rayvan Production",
    externalAccountName: "Rayvan Production",
    status: "connected",
    uiStatus: "attention_required",
    fields: [
      { label: "Project", value: "desktop" },
      { label: "Environment", value: "production" },
      { label: "Unresolved issues", value: "2" },
    ],
    grantPermissions: true,
  },
  {
    pluginId: "example-local",
    connectionName: "Local Dev",
    externalAccountName: "Local Dev",
    status: "connected",
    fields: [
      { label: "Service", value: "local-dev-service" },
      { label: "Status", value: "Running" },
    ],
    grantPermissions: false,
  },
];

/**
 * Seeds the canonical demo connections for a project, using already-installed
 * catalog records. Intended to run once per project id (callers are
 * responsible for tracking which projects have been seeded).
 */
export async function seedProjectConnections(
  deps: {
    connections: PluginConnectionRepository;
    permissionGrants: PluginPermissionGrantRepository;
  },
  projectId: string,
  installedByPluginId: Map<string, InstalledPluginRecord>,
): Promise<void> {
  const now = new Date().toISOString();

  for (const spec of SEED_CONNECTIONS) {
    const installed = installedByPluginId.get(spec.pluginId);
    if (!installed) {
      continue;
    }

    const connectionId = crypto.randomUUID();
    await deps.connections.save({
      id: connectionId,
      installedPluginId: installed.id,
      pluginId: installed.pluginId,
      projectId,
      name: spec.connectionName,
      status: spec.status,
      externalAccountName: spec.externalAccountName,
      metadata: {
        uiCard: { fields: spec.fields },
        uiDetail: { fields: spec.fields },
        ...(spec.uiStatus ? { uiStatus: spec.uiStatus } : {}),
      },
      schemaVersion: "1",
      createdAt: now,
      updatedAt: now,
      lastAuthenticatedAt: now,
      lastSuccessfulSyncAt: now,
    });

    if (spec.grantPermissions) {
      await deps.permissionGrants.save({
        id: crypto.randomUUID(),
        pluginId: installed.pluginId,
        connectionId,
        permission: "network",
        projectId,
        granted: true,
        grantedBy: DEV_FIXTURE_SYSTEM_ACTOR,
        grantedAt: now,
        reason: "Seeded development fixture",
      });
    }
  }
}
