import type {
  InstalledPluginRecord,
  PluginConnectionRecord,
  PluginPermissionGrantRecord,
} from "@rayvan/local-database";
import type { PluginManifest } from "@rayvan/plugin-sdk";
import { describe, expect, it } from "vitest";

import {
  mapConnectionToCardViewModel,
  mapConnectionToDetailViewModel,
  mapIconFromManifest,
  mapInstalledPluginToLibraryViewModel,
  resolveIntegrationStatus,
} from "./mappers.js";

const BASE_MANIFEST: PluginManifest = {
  id: "vercel",
  name: "Vercel",
  description: "Connect Rayvan to Vercel projects and deployments.",
  version: "0.0.1",
  publisher: "rayvan",
  rayvanApiVersion: "1",
  capabilities: [],
  permissions: [],
  resourceTypes: [],
  presentation: {
    icon: { iconId: "vercel", initials: "V", label: "Vercel" },
    theme: { surface: "dark", accentColor: "#FFFFFF", foregroundMode: "light" },
    supportsMultipleConnections: true,
  },
};

function buildInstalled(overrides: Partial<InstalledPluginRecord> = {}): InstalledPluginRecord {
  return {
    id: "installed-1",
    pluginId: "vercel",
    pluginVersion: "0.0.1",
    manifestVersion: "0.0.1",
    rayvanApiVersion: "1",
    source: { type: "built_in" },
    status: "installed",
    enabled: true,
    installedAt: "2026-01-01T00:00:00.000Z",
    updatedAt: "2026-01-01T00:00:00.000Z",
    manifestSnapshot: BASE_MANIFEST,
    ...overrides,
  };
}

function buildConnection(
  overrides: Partial<PluginConnectionRecord> = {},
): PluginConnectionRecord {
  return {
    id: "connection-1",
    installedPluginId: "installed-1",
    pluginId: "vercel",
    projectId: "project-1",
    name: "Acme Organisation",
    status: "connected",
    metadata: {},
    schemaVersion: "1",
    createdAt: "2026-01-01T00:00:00.000Z",
    updatedAt: "2026-01-01T00:00:00.000Z",
    ...overrides,
  };
}

describe("resolveIntegrationStatus", () => {
  it("maps raw connection statuses to UI statuses", () => {
    expect(resolveIntegrationStatus(buildConnection({ status: "connected" })).status).toBe(
      "connected",
    );
    expect(resolveIntegrationStatus(buildConnection({ status: "pending" })).status).toBe(
      "syncing",
    );
    expect(resolveIntegrationStatus(buildConnection({ status: "expired" })).status).toBe(
      "expired",
    );
    expect(resolveIntegrationStatus(buildConnection({ status: "revoked" })).status).toBe(
      "disconnected",
    );
    expect(resolveIntegrationStatus(buildConnection({ status: "disconnected" })).status).toBe(
      "disconnected",
    );
    expect(resolveIntegrationStatus(buildConnection({ status: "error" })).status).toBe("error");
  });

  it("honors a metadata.uiStatus override", () => {
    const connection = buildConnection({
      status: "connected",
      metadata: { uiStatus: "attention_required" },
    });
    const resolved = resolveIntegrationStatus(connection);
    expect(resolved.status).toBe("attention_required");
    expect(resolved.label).toBe("Attention required");
  });
});

describe("mapIconFromManifest", () => {
  it("falls back to manifest name initials when presentation.icon is missing", () => {
    const manifest: PluginManifest = { ...BASE_MANIFEST, presentation: undefined };
    const icon = mapIconFromManifest(manifest);
    expect(icon.initials).toBe("VE");
    expect(icon.label).toBe("Vercel");
  });
});

describe("mapConnectionToCardViewModel", () => {
  it("reads card fields from connection.metadata.uiCard", () => {
    const connection = buildConnection({
      metadata: {
        uiCard: {
          fields: [
            { label: "Project", value: "rayvan-web" },
            { label: "Environment", value: "Production" },
          ],
        },
      },
    });
    const card = mapConnectionToCardViewModel(connection, buildInstalled());

    expect(card.connectionName).toBe("Acme Organisation");
    expect(card.pluginName).toBe("Vercel");
    expect(card.fields).toEqual([
      { label: "Project", value: "rayvan-web" },
      { label: "Environment", value: "Production" },
    ]);
    expect(card.status).toBe("connected");
    expect(card.actions.map((action) => action.id)).toEqual(["open", "sync", "configure"]);
  });

  it("returns no fields when metadata.uiCard is absent", () => {
    const card = mapConnectionToCardViewModel(buildConnection(), buildInstalled());
    expect(card.fields).toEqual([]);
  });
});

describe("mapConnectionToDetailViewModel", () => {
  it("builds overview groups from connection state, metadata, and grants", () => {
    const connection = buildConnection({
      metadata: {
        uiDetail: { fields: [{ label: "Project", value: "rayvan-web" }] },
      },
      lastSuccessfulSyncAt: "2026-02-01T00:00:00.000Z",
    });
    const grants: PluginPermissionGrantRecord[] = [
      {
        id: "grant-1",
        pluginId: "vercel",
        connectionId: connection.id,
        permission: "network",
        granted: true,
        grantedBy: { type: "user", id: "user-1" },
        grantedAt: "2026-01-01T00:00:00.000Z",
      },
    ];

    const detail = mapConnectionToDetailViewModel(connection, buildInstalled(), grants);

    expect(detail.overview.groups.map((group) => group.title)).toEqual([
      "Connection",
      "Vercel details",
      "Permissions",
    ]);
    expect(detail.configuration.grants).toEqual([
      { permission: "network", granted: true, grantedAt: "2026-01-01T00:00:00.000Z" },
    ]);
    expect(detail.resources.isEmpty).toBe(true);
    expect(detail.activity.isEmpty).toBe(true);
  });
});

describe("mapInstalledPluginToLibraryViewModel", () => {
  it("marks multi-connection plugins eligible even when already configured", () => {
    const installed = buildInstalled();
    const connections = [buildConnection({ installedPluginId: installed.id })];
    const library = mapInstalledPluginToLibraryViewModel(installed, connections);

    expect(library.supportsMultipleConnections).toBe(true);
    expect(library.existingConnectionCount).toBe(1);
    expect(library.eligible).toBe(true);
    expect(library.badge).toBe("built-in");
  });

  it("marks single-connection plugins ineligible once configured", () => {
    const manifest: PluginManifest = {
      ...BASE_MANIFEST,
      id: "example-local",
      name: "Example Local",
      presentation: {
        ...BASE_MANIFEST.presentation,
        supportsMultipleConnections: false,
      },
    };
    const installed = buildInstalled({
      id: "installed-2",
      pluginId: "example-local",
      manifestSnapshot: manifest,
    });
    const connections = [
      buildConnection({ installedPluginId: installed.id, pluginId: "example-local" }),
    ];
    const library = mapInstalledPluginToLibraryViewModel(installed, connections);

    expect(library.supportsMultipleConnections).toBe(false);
    expect(library.existingConnectionCount).toBe(1);
    expect(library.eligible).toBe(false);
  });

  it("marks single-connection plugins eligible when not yet configured", () => {
    const manifest: PluginManifest = {
      ...BASE_MANIFEST,
      presentation: { ...BASE_MANIFEST.presentation, supportsMultipleConnections: false },
    };
    const installed = buildInstalled({ manifestSnapshot: manifest });
    const library = mapInstalledPluginToLibraryViewModel(installed, []);
    expect(library.eligible).toBe(true);
  });

  it("defaults supportsMultipleConnections to false when omitted", () => {
    const installed = buildInstalled({
      manifestSnapshot: {
        ...BASE_MANIFEST,
        presentation: {
          icon: BASE_MANIFEST.presentation?.icon,
          theme: BASE_MANIFEST.presentation?.theme,
        },
      },
    });
    const connections = [buildConnection({ installedPluginId: installed.id })];
    const library = mapInstalledPluginToLibraryViewModel(installed, connections);
    expect(library.supportsMultipleConnections).toBe(false);
    expect(library.eligible).toBe(false);
  });

  it("marks third-party plugins with no badge", () => {
    const installed = buildInstalled({
      source: { type: "package", packageId: "runpod-community-plugin" },
      manifestSnapshot: { ...BASE_MANIFEST, publisher: "runpod-community" },
    });
    const library = mapInstalledPluginToLibraryViewModel(installed, []);
    expect(library.badge).toBeNull();
  });
});
