import { describe, expect, it } from "vitest";
import { environmentId, projectId, type Environment } from "@rayvan/core";
import type { ConfigurationMatrixViewModel } from "@rayvan/config-engine";

import {
  mapComparisonSummary,
  mapEnvironmentToCardViewModel,
  mapMatrixHealthByEnvironment,
  mapResourcesViewModel,
} from "./mappers.js";

const environment: Environment = {
  id: environmentId("env-1"),
  projectId: projectId("project-1"),
  name: "Development",
  slug: "development",
  kind: "development",
  status: "healthy",
  presentation: { color: "blue", icon: "development" },
  createdAt: "2026-01-01T00:00:00.000Z",
  updatedAt: "2026-01-01T00:00:00.000Z",
};

const matrix: ConfigurationMatrixViewModel = {
  projectId: "project-1",
  columns: [
    {
      environmentId: "env-1",
      name: "Development",
      slug: "development",
      kind: "development",
    },
  ],
  rows: [
    {
      configurationKeyId: "key-1",
      name: "API_BASE_URL",
      required: true,
      sensitive: false,
      valueType: "url",
      cells: [
        {
          configurationKeyId: "key-1",
          configurationKeyName: "API_BASE_URL",
          environmentId: "env-1",
          status: "healthy",
          statusLabel: "Healthy",
          occurrenceCount: 1,
          accessLocked: false,
          requiredMissing: false,
          warningCount: 0,
          occurrenceIds: ["occ-1"],
        },
      ],
    },
    {
      configurationKeyId: "key-2",
      name: "STRIPE_SECRET_KEY",
      required: true,
      sensitive: true,
      valueType: "secret",
      cells: [
        {
          configurationKeyId: "key-2",
          configurationKeyName: "STRIPE_SECRET_KEY",
          environmentId: "env-1",
          status: "missing",
          statusLabel: "Missing",
          occurrenceCount: 0,
          accessLocked: false,
          requiredMissing: true,
          warningCount: 1,
          occurrenceIds: [],
        },
      ],
    },
  ],
  summary: {
    keyCount: 2,
    environmentCount: 1,
    missingCellCount: 1,
    mismatchedCellCount: 0,
    lockedCellCount: 0,
    healthyCellCount: 1,
  },
};

describe("environment mappers", () => {
  it("maps matrix health and comparison summaries", () => {
    const health = mapMatrixHealthByEnvironment(matrix).get("env-1");
    expect(health).toEqual({
      healthy: 1,
      missing: 1,
      mismatched: 0,
      locked: 0,
    });
    expect(mapComparisonSummary(matrix).missingCellCount).toBe(1);
  });

  it("maps environment cards with status labels and counts", () => {
    const card = mapEnvironmentToCardViewModel({
      environment,
      bindings: [
        {
          id: "bind-1",
          projectId: "project-1",
          environmentId: "env-1",
          discoveredResourceId: "res-1",
          pluginId: "vercel",
          connectionId: "conn-1",
          bindingStatus: "active",
          createdBy: { type: "system", id: "test" },
          createdAt: "2026-01-01T00:00:00.000Z",
          updatedAt: "2026-01-01T00:00:00.000Z",
        },
      ],
      resources: [],
      findings: [
        {
          id: "finding-1",
          projectId: "project-1",
          environmentId: "env-1",
          severity: "error",
          category: "missing_configuration",
          title: "Missing",
          description: "Missing key",
        },
      ],
      matrix,
      syncState: null,
      keyCount: 2,
    });

    expect(card.statusLabel).toBe("Healthy");
    expect(card.resourceCount).toBe(1);
    expect(card.integrationCount).toBe(1);
    expect(card.findingsCount).toBe(1);
    expect(card.health.missing).toBe(1);
  });

  it("groups bound resources and keeps unmapped separate", () => {
    const viewModel = mapResourcesViewModel({
      environments: [environment],
      resources: [
        {
          id: "res-1",
          pluginId: "vercel",
          installedPluginId: "inst-1",
          connectionId: "conn-1",
          providerResourceId: "env:development",
          resourceType: "environment",
          name: "Vercel Development",
          metadata: {},
          pluginVersion: "1.0.0",
          schemaVersion: "1",
          discoveryStatus: "active",
          firstDiscoveredAt: "2026-01-01T00:00:00.000Z",
          lastDiscoveredAt: "2026-01-01T00:00:00.000Z",
        },
        {
          id: "res-2",
          pluginId: "github",
          installedPluginId: "inst-2",
          connectionId: "conn-2",
          providerResourceId: "branch:develop",
          resourceType: "branch",
          name: "GitHub develop",
          metadata: {},
          pluginVersion: "1.0.0",
          schemaVersion: "1",
          discoveryStatus: "active",
          firstDiscoveredAt: "2026-01-01T00:00:00.000Z",
          lastDiscoveredAt: "2026-01-01T00:00:00.000Z",
        },
      ],
      bindings: [
        {
          id: "bind-1",
          projectId: "project-1",
          environmentId: "env-1",
          discoveredResourceId: "res-1",
          pluginId: "vercel",
          connectionId: "conn-1",
          bindingStatus: "active",
          createdBy: { type: "system", id: "test" },
          createdAt: "2026-01-01T00:00:00.000Z",
          updatedAt: "2026-01-01T00:00:00.000Z",
        },
      ],
    });

    expect(viewModel.groups[0]?.title).toBe("Development");
    expect(viewModel.groups[0]?.items).toHaveLength(1);
    expect(viewModel.groups.at(-1)?.title).toBe("Unmapped");
    expect(viewModel.groups.at(-1)?.items[0]?.name).toBe("GitHub develop");
  });

  it("maps config aggregate and review action from status service", () => {
    const card = mapEnvironmentToCardViewModel({
      environment,
      bindings: [],
      resources: [],
      findings: [],
      matrix: null,
      syncState: null,
      keyCount: 3,
      configStatus: {
        environmentId: "env-1",
        keyStatuses: [],
        summary: {
          inSyncCount: 2,
          localChangesCount: 1,
          remoteChangedCount: 0,
          mismatchedCount: 0,
          missingRemoteCount: 1,
          missingLocalCount: 0,
          notManagedCount: 0,
          partiallyAppliedCount: 0,
          lockedCount: 0,
          unknownCount: 0,
          unsavedDraftCount: 0,
          staleObservedCount: 0,
        },
        headlineLabel: "Changes not applied",
        hasUnsavedLocalChanges: false,
        hasChangesNotApplied: true,
      },
    });

    expect(card.configAggregate?.headlineLabel).toBe("Changes not applied");
    expect(card.configAggregate?.changesNotAppliedCount).toBe(1);
    expect(card.configAggregate?.missingRemoteCount).toBe(1);
    expect(card.actions.some((action) => action.id === "review_changes")).toBe(
      true,
    );
  });
});
