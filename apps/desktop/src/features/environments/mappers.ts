import type { Environment } from "@rayvan/core";
import type {
  ConfigurationDerivedFinding,
  ConfigurationMatrixViewModel,
} from "@rayvan/config-engine";
import type {
  DiscoveredResourceRecord,
  EnvironmentMappingSuggestionRecord,
  ResourceBindingRecord,
} from "@rayvan/local-database";
import type { EnvironmentSyncState } from "../../lib/environments/index.js";

import {
  ENVIRONMENT_KIND_LABELS,
  ENVIRONMENT_STATUS_LABELS,
  type EnvironmentCardViewModel,
  type EnvironmentComparisonSummary,
  type EnvironmentHealthSummary,
  type EnvironmentResourcesViewModel,
  type MappingSuggestionViewModel,
  type ResourceListItemViewModel,
} from "./view-models.js";

function emptyHealth(): EnvironmentHealthSummary {
  return { healthy: 0, missing: 0, mismatched: 0, locked: 0 };
}

export function mapMatrixHealthByEnvironment(
  matrix: ConfigurationMatrixViewModel | null,
): Map<string, EnvironmentHealthSummary> {
  const map = new Map<string, EnvironmentHealthSummary>();
  if (!matrix) {
    return map;
  }
  for (const column of matrix.columns) {
    map.set(column.environmentId, emptyHealth());
  }
  for (const row of matrix.rows) {
    for (const cell of row.cells) {
      const health = map.get(cell.environmentId) ?? emptyHealth();
      switch (cell.status) {
        case "healthy":
          health.healthy += 1;
          break;
        case "missing":
          health.missing += 1;
          break;
        case "mismatched":
          health.mismatched += 1;
          break;
        case "locked":
          health.locked += 1;
          break;
        default:
          break;
      }
      map.set(cell.environmentId, health);
    }
  }
  return map;
}

export function mapEnvironmentToCardViewModel(input: {
  environment: Environment;
  bindings: ResourceBindingRecord[];
  resources: DiscoveredResourceRecord[];
  findings: ConfigurationDerivedFinding[];
  matrix: ConfigurationMatrixViewModel | null;
  syncState: EnvironmentSyncState | null;
  keyCount: number;
  configStatus?: import("@rayvan/config-engine").EnvironmentConfigurationStatusViewModel | null;
}): EnvironmentCardViewModel {
  const { environment } = input;
  const activeBindings = input.bindings.filter(
    (binding) =>
      binding.environmentId === environment.id && binding.bindingStatus === "active",
  );
  const pluginIds = new Set(activeBindings.map((binding) => binding.pluginId));
  const findingsCount = input.findings.filter(
    (finding) => finding.environmentId === environment.id,
  ).length;
  const health =
    mapMatrixHealthByEnvironment(input.matrix).get(environment.id) ?? emptyHealth();

  const lastSyncLabel =
    input.syncState?.lastResult?.finishedAt &&
    (!input.syncState.lastResult.environmentId ||
      input.syncState.lastResult.environmentId === environment.id)
      ? new Date(input.syncState.lastResult.finishedAt).toLocaleString()
      : environment.status === "local_only"
        ? "Never synced"
        : "Not synced in this session";

  const configAggregate = input.configStatus
    ? {
        headlineLabel: input.configStatus.headlineLabel,
        inSyncCount: input.configStatus.summary.inSyncCount,
        changesNotAppliedCount: input.configStatus.summary.localChangesCount,
        missingRemoteCount: input.configStatus.summary.missingRemoteCount,
        missingLocalCount:
          input.configStatus.summary.missingLocalCount +
          input.configStatus.summary.notManagedCount,
        remoteChangedCount: input.configStatus.summary.remoteChangedCount,
        lockedCount: input.configStatus.summary.lockedCount,
        hasUnsavedLocalChanges: input.configStatus.hasUnsavedLocalChanges,
        hasChangesNotApplied: input.configStatus.hasChangesNotApplied,
      }
    : undefined;

  const actions: EnvironmentCardViewModel["actions"] = [
    { id: "open", label: "Open", kind: "primary" },
  ];
  if (configAggregate?.hasChangesNotApplied) {
    actions.push({
      id: "review_changes",
      label: "Review changes",
      kind: "secondary",
    });
  }
  actions.push(
    { id: "sync", label: "Sync", kind: "secondary" },
    { id: "edit", label: "Edit", kind: "overflow" },
    { id: "archive", label: "Archive", kind: "overflow" },
  );

  return {
    environmentId: environment.id,
    name: environment.name,
    kind: environment.kind,
    kindLabel: ENVIRONMENT_KIND_LABELS[environment.kind],
    status: environment.status,
    statusLabel: ENVIRONMENT_STATUS_LABELS[environment.status],
    description: environment.description,
    color: environment.presentation?.color,
    icon: environment.presentation?.icon,
    resourceCount: activeBindings.length,
    integrationCount: pluginIds.size,
    configurationKeyCount: input.keyCount,
    findingsCount,
    lastSyncLabel,
    health,
    configAggregate,
    actions,
  };
}

export function mapComparisonSummary(
  matrix: ConfigurationMatrixViewModel | null,
): EnvironmentComparisonSummary {
  if (!matrix) {
    return {
      environmentCount: 0,
      keyCount: 0,
      missingCellCount: 0,
      mismatchedCellCount: 0,
      healthyCellCount: 0,
    };
  }
  return {
    environmentCount: matrix.summary.environmentCount,
    keyCount: matrix.summary.keyCount,
    missingCellCount: matrix.summary.missingCellCount,
    mismatchedCellCount: matrix.summary.mismatchedCellCount,
    healthyCellCount: matrix.summary.healthyCellCount,
  };
}

export function mapSuggestionsToViewModels(
  suggestions: EnvironmentMappingSuggestionRecord[],
  resources: DiscoveredResourceRecord[],
): MappingSuggestionViewModel[] {
  const resourceById = new Map(resources.map((resource) => [resource.id, resource]));
  return suggestions.map((suggestion) => {
    const resource = resourceById.get(suggestion.discoveredResourceId);
    return {
      suggestionId: suggestion.id,
      resourceName: resource?.name ?? "Unknown resource",
      pluginId: resource?.pluginId ?? suggestion.connectionId,
      suggestedEnvironmentId: suggestion.suggestedEnvironmentId,
      suggestedEnvironmentName: suggestion.suggestedEnvironmentName,
      confidence: suggestion.confidence,
      reasons: suggestion.reasons,
    };
  });
}

export function mapResourcesViewModel(input: {
  environments: Environment[];
  resources: DiscoveredResourceRecord[];
  bindings: ResourceBindingRecord[];
}): EnvironmentResourcesViewModel {
  const envById = new Map<string, Environment>(
    input.environments.map((env) => [env.id, env]),
  );
  const activeBindings = input.bindings.filter(
    (binding) => binding.bindingStatus === "active",
  );
  const bindingByResource = new Map(
    activeBindings.map((binding) => [binding.discoveredResourceId, binding]),
  );

  const groups: EnvironmentResourcesViewModel["groups"] = input.environments.map(
    (environment) => ({
      environmentId: environment.id,
      title: environment.name,
      items: [] as ResourceListItemViewModel[],
    }),
  );
  const unmapped: ResourceListItemViewModel[] = [];

  for (const resource of input.resources) {
    const binding = bindingByResource.get(resource.id);
    const item: ResourceListItemViewModel = {
      discoveredResourceId: resource.id,
      bindingId: binding?.id,
      name: resource.name,
      pluginId: resource.pluginId,
      resourceType: resource.resourceType,
      environmentId: binding?.environmentId,
      environmentName: binding?.environmentId
        ? envById.get(binding.environmentId)?.name
        : undefined,
      bindingStatus: binding?.bindingStatus,
    };

    if (binding?.environmentId) {
      const group = groups.find((entry) => entry.environmentId === binding.environmentId);
      group?.items.push(item);
    } else {
      unmapped.push(item);
    }
  }

  return {
    groups: [
      ...groups.filter((group) => group.items.length > 0),
      { environmentId: null, title: "Unmapped", items: unmapped },
    ],
  };
}
