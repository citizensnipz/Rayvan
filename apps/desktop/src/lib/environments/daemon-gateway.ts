import type {
  AppliedConfigurationState,
  ConfigurationKey,
  ConfigurationOccurrence,
  DesiredConfigurationValue,
  Environment,
  FindingRecord,
  FindingSummary,
} from "@rayvan/core";
import type {
  ConfigurationApplyPlan,
  ConfigurationApplyResult,
  ConfigurationMatrixViewModel,
  EnvironmentConfigurationStatusViewModel,
} from "@rayvan/config-engine";
import type {
  DiscoveredResourceRecord,
  EnvironmentMappingSuggestionRecord,
  ResourceBindingRecord,
} from "@rayvan/local-database";

import { daemonRequest, desktopDaemon } from "../daemon/client.js";
import type {
  AcceptSuggestionInput,
  AcceptSuggestionResult,
  AdoptDiscoveredKeyGatewayInput,
  CreateEnvironmentGatewayInput,
  EnvironmentsGateway,
  EnvironmentSyncResult,
  EnvironmentSyncState,
  SaveDesiredValueGatewayInput,
  UpdateConfigurationKeyGatewayInput,
  UpdateEnvironmentGatewayInput,
} from "./types.js";

function unsupported(method: string): never {
  throw new Error(
    `Daemon environments gateway does not implement ${method} yet`,
  );
}

function emptyMatrix(projectId: string): ConfigurationMatrixViewModel {
  return {
    projectId,
    columns: [],
    rows: [],
    summary: {
      keyCount: 0,
      environmentCount: 0,
      missingCellCount: 0,
      mismatchedCellCount: 0,
      lockedCellCount: 0,
      healthyCellCount: 0,
    },
  };
}

function emptyConfigStatus(
  environmentId: string,
): EnvironmentConfigurationStatusViewModel {
  return {
    environmentId,
    keyStatuses: [],
    summary: {
      inSyncCount: 0,
      localChangesCount: 0,
      remoteChangedCount: 0,
      mismatchedCount: 0,
      missingRemoteCount: 0,
      missingLocalCount: 0,
      notManagedCount: 0,
      partiallyAppliedCount: 0,
      lockedCount: 0,
      unknownCount: 0,
      unsavedDraftCount: 0,
      staleObservedCount: 0,
    },
    headlineLabel: "No configuration status from daemon yet",
    hasUnsavedLocalChanges: false,
    hasChangesNotApplied: false,
  };
}

function idleSyncState(projectId: string): EnvironmentSyncState {
  return {
    projectId,
    inProgress: false,
    phase: "idle",
    cancelled: false,
  };
}

/**
 * Daemon-backed environments gateway. CRUD + configuration reads go to
 * `rayvand`; matrix/mapping helpers that lack daemon methods return empty
 * structures so the workspace can still load.
 */
export function createDaemonEnvironmentsGateway(): EnvironmentsGateway {
  const syncByProject = new Map<string, EnvironmentSyncState>();

  return {
    async ensureProjectSeeded(): Promise<void> {},

    async listEnvironments(
      projectId: string,
      options?: { includeArchived?: boolean },
    ): Promise<Environment[]> {
      return (await desktopDaemon.listEnvironments(
        projectId,
        options,
      )) as Environment[];
    },

    async createEnvironment(
      input: CreateEnvironmentGatewayInput,
    ): Promise<Environment> {
      return (await desktopDaemon.createEnvironment({
        projectId: input.projectId,
        name: input.name,
        slug: input.slug,
        kind: input.kind,
        description: input.description,
      })) as Environment;
    },

    async updateEnvironment(
      id: string,
      input: UpdateEnvironmentGatewayInput,
    ): Promise<Environment> {
      return (await desktopDaemon.updateEnvironment(
        id,
        input as Record<string, unknown>,
      )) as Environment;
    },

    async archiveEnvironment(id: string): Promise<Environment> {
      return (await desktopDaemon.archiveEnvironment(id)) as Environment;
    },

    async getEnvironment(id: string): Promise<Environment | null> {
      return (await desktopDaemonRequest("environments.get", {
        environmentId: id,
      })) as Environment | null;
    },

    async listConfigurationKeys(projectId: string): Promise<ConfigurationKey[]> {
      return (await desktopDaemon.listConfigurationKeys(
        projectId,
      )) as ConfigurationKey[];
    },

    async updateConfigurationKey(
      id: string,
      metadata: UpdateConfigurationKeyGatewayInput,
    ): Promise<ConfigurationKey> {
      return (await desktopDaemonRequest("configuration.setMetadata", {
        configurationKeyId: id,
        ...metadata,
      })) as ConfigurationKey;
    },

    async listOccurrences(projectId: string): Promise<ConfigurationOccurrence[]> {
      const result = await desktopDaemonRequest<{
        occurrences?: ConfigurationOccurrence[];
      }>("environments.getResources", { projectId }).catch(() => null);
      return result?.occurrences ?? [];
    },

    async listDesiredValues(projectId: string): Promise<DesiredConfigurationValue[]> {
      void projectId;
      return [];
    },

    async listDesiredValuesByEnvironment(
      environmentId: string,
    ): Promise<DesiredConfigurationValue[]> {
      void environmentId;
      return [];
    },

    async listAppliedByEnvironment(
      environmentId: string,
    ): Promise<AppliedConfigurationState[]> {
      void environmentId;
      return [];
    },

    async saveDesiredValues(
      inputs: SaveDesiredValueGatewayInput[],
    ): Promise<DesiredConfigurationValue[]> {
      const saved: DesiredConfigurationValue[] = [];
      for (const input of inputs) {
        if (input.desiredValue !== undefined) {
          const result = (await desktopDaemon.setConfigurationValue({
            configurationKeyId: input.configurationKeyId,
            environmentId: input.environmentId,
            projectId: input.projectId,
            value: input.desiredValue,
            expectedRevision: input.expectedRevision,
          })) as DesiredConfigurationValue;
          saved.push(result);
        } else if (input.secretValueRef !== undefined) {
          const result = (await desktopDaemon.setSensitiveConfigurationValue({
            configurationKeyId: input.configurationKeyId,
            environmentId: input.environmentId,
            projectId: input.projectId,
            value: input.secretValueRef,
            expectedRevision: input.expectedRevision,
          })) as DesiredConfigurationValue;
          saved.push(result);
        }
      }
      return saved;
    },

    async getEnvironmentConfigurationStatus(
      projectId: string,
      environmentId: string,
    ): Promise<EnvironmentConfigurationStatusViewModel> {
      void projectId;
      try {
        await desktopDaemon.getEnvironmentConfiguration(projectId, environmentId);
      } catch {
        // Fall through to empty status — workspace still loads.
      }
      return emptyConfigStatus(environmentId);
    },

    async buildApplyPlan(): Promise<ConfigurationApplyPlan> {
      unsupported("buildApplyPlan");
    },

    async approveApplyPlan(): Promise<ConfigurationApplyResult> {
      unsupported("approveApplyPlan");
    },

    async adoptDiscoveredKey(
      input: AdoptDiscoveredKeyGatewayInput,
    ): Promise<DesiredConfigurationValue> {
      return (await desktopDaemonRequest(
        "configuration.adoptDiscovered",
        input,
      )) as DesiredConfigurationValue;
    },

    async listDiscoveredResources(): Promise<DiscoveredResourceRecord[]> {
      return [];
    },

    async listBindings(): Promise<ResourceBindingRecord[]> {
      return [];
    },

    async attachResource(): Promise<ResourceBindingRecord> {
      unsupported("attachResource");
    },

    async moveResource(): Promise<ResourceBindingRecord> {
      unsupported("moveResource");
    },

    async detachResource(): Promise<ResourceBindingRecord> {
      unsupported("detachResource");
    },

    async listPendingSuggestions(): Promise<EnvironmentMappingSuggestionRecord[]> {
      return [];
    },

    async acceptSuggestion(
      input: AcceptSuggestionInput,
    ): Promise<AcceptSuggestionResult> {
      void input;
      unsupported("acceptSuggestion");
    },

    async rejectSuggestion(): Promise<EnvironmentMappingSuggestionRecord> {
      unsupported("rejectSuggestion");
    },

    async chooseSuggestionEnvironment(): Promise<EnvironmentMappingSuggestionRecord> {
      unsupported("chooseSuggestionEnvironment");
    },

    async getMatrix(projectId: string): Promise<ConfigurationMatrixViewModel> {
      return emptyMatrix(projectId);
    },

    async listOpenFindings(projectId: string): Promise<FindingRecord[]> {
      return (await desktopDaemon.listFindings({
        projectId,
        status: "open",
      })) as FindingRecord[];
    },

    async getFindingsSummary(projectId: string): Promise<FindingSummary> {
      const raw = await desktopDaemon.getFindingSummary({ projectId });
      return raw as FindingSummary;
    },

    async getEnvironmentFindingsSummary(
      projectId: string,
      environmentId: string,
    ): Promise<FindingSummary> {
      const raw = await desktopDaemon.getFindingSummary({
        projectId,
        environmentId,
      });
      return raw as FindingSummary;
    },

    async syncWithIntegrations(
      projectId: string,
      options?: { environmentId?: string },
    ): Promise<EnvironmentSyncResult> {
      syncByProject.set(projectId, {
        projectId,
        inProgress: true,
        phase: "discovering",
        cancelled: false,
        environmentId: options?.environmentId,
      });
      try {
        if (options?.environmentId) {
          await desktopDaemonRequest("integrations.syncEnvironment", {
            environmentId: options.environmentId,
          });
        } else {
          await desktopDaemon.syncProject(projectId);
        }
        const result: EnvironmentSyncResult = {
          projectId,
          environmentId: options?.environmentId,
          phase: "complete",
          cancelled: false,
          startedAt: new Date().toISOString(),
          finishedAt: new Date().toISOString(),
          plugins: [],
          suggestionsCreated: 0,
          resourcesTouched: 0,
        };
        syncByProject.set(projectId, {
          projectId,
          inProgress: false,
          phase: "complete",
          cancelled: false,
          lastResult: result,
          environmentId: options?.environmentId,
        });
        return result;
      } catch (error) {
        syncByProject.set(projectId, {
          projectId,
          inProgress: false,
          phase: "failed",
          cancelled: false,
          environmentId: options?.environmentId,
        });
        throw error;
      }
    },

    async cancelSync(projectId: string): Promise<void> {
      const current = syncByProject.get(projectId);
      if (current) {
        syncByProject.set(projectId, {
          ...current,
          inProgress: false,
          phase: "cancelled",
          cancelled: true,
        });
      }
    },

    async getSyncState(projectId: string): Promise<EnvironmentSyncState> {
      return syncByProject.get(projectId) ?? idleSyncState(projectId);
    },
  };
}

async function desktopDaemonRequest<T = unknown>(
  method: string,
  params?: unknown,
): Promise<T> {
  return daemonRequest<T>(method, params);
}
