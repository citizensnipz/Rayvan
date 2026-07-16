import {
  APPLY_EFFECT_WARNINGS,
  buildConfigurationMatrix,
  deriveEnvironmentStatus,
  type ConfigurationApplyPlan,
  type ConfigurationApplyResult,
} from "@rayvan/config-engine";
import type { DesiredConfigurationValue, Environment } from "@rayvan/core";
import {
  ConfigurationDesiredStateService,
  ConfigurationService,
  EnvironmentMappingService,
  EnvironmentService,
  ResourceBindingService,
  ResourceDiscoveryService,
  createInMemoryEnvironmentPersistence,
  createInMemoryPluginPersistence,
} from "@rayvan/local-database";
import type {
  DiscoveredResourceRecord,
  EnvironmentMappingSuggestionRecord,
  PluginConnectionRecord,
  ResourceBindingRecord,
} from "@rayvan/local-database";

import { createDevFindingsGateway } from "../findings/index.js";
import { DEV_FIXTURE_USER_ACTOR } from "../plugins/dev-fixtures.js";
import { seedProjectEnvironments } from "./dev-fixtures.js";
import type {
  AcceptSuggestionInput,
  AcceptSuggestionResult,
  AdoptDiscoveredKeyGatewayInput,
  AttachResourceInput,
  CreateEnvironmentGatewayInput,
  EnvironmentSyncPhase,
  EnvironmentSyncPluginResult,
  EnvironmentSyncResult,
  EnvironmentSyncState,
  EnvironmentsGateway,
  MoveResourceInput,
  SaveDesiredValueGatewayInput,
  UpdateConfigurationKeyGatewayInput,
  UpdateEnvironmentGatewayInput,
} from "./types.js";

const DESKTOP_USER_ACTOR = DEV_FIXTURE_USER_ACTOR;
const DESKTOP_CONFIG_ACTOR = {
  kind: "user" as const,
  id: DEV_FIXTURE_USER_ACTOR.id,
  displayName:
    DEV_FIXTURE_USER_ACTOR.type === "user"
      ? DEV_FIXTURE_USER_ACTOR.displayName
      : undefined,
};

interface SyncControl {
  cancelRequested: boolean;
  state: EnvironmentSyncState;
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
 * DEVELOPMENT ONLY gateway for the Environments workspace.
 * Fresh isolated instance per call — no module-level singleton.
 */
export function createDevEnvironmentsGateway(): EnvironmentsGateway {
  const envPersistence = createInMemoryEnvironmentPersistence();
  const pluginPersistence = createInMemoryPluginPersistence();
  /** Shared findings store for this gateway instance (persisted FindingRecords). */
  const findingsGateway = createDevFindingsGateway();

  const environmentService = new EnvironmentService(envPersistence.environments);
  const configurationService = new ConfigurationService(
    envPersistence.configuration.keys,
    envPersistence.configuration.occurrences,
  );
  const desiredStateService =
    envPersistence.configuration.desiredStateService ??
    new ConfigurationDesiredStateService(
      envPersistence.configuration.keys,
      envPersistence.configuration.desired,
      envPersistence.configuration.applied,
    );
  const bindingService = new ResourceBindingService(
    pluginPersistence.discoveredResources,
    pluginPersistence.resourceBindings,
  );
  const mappingService = new EnvironmentMappingService(
    pluginPersistence.mappingSuggestions,
  );
  const discoveryService = new ResourceDiscoveryService(
    pluginPersistence.connections,
    pluginPersistence.discoveredResources,
  );

  /**
   * In-flight / completed seed promises per project.
   * Concurrent callers must await the same promise — never treat a project as
   * seeded before the seed finishes (React Strict Mode / effect races).
   */
  const seedByProject = new Map<string, Promise<void>>();
  const syncByProject = new Map<string, SyncControl>();
  /** Simulates a one-time Sentry discovery failure per gateway instance. */
  let sentryFailOnce = true;
  /** Pending apply plans (stub — no provider execution). */
  const applyPlans = new Map<string, ConfigurationApplyPlan>();
  /** Deterministic stub: first approve for Production marks one op failed. */
  let simulatePartialApplyOnce = true;

  function getSyncControl(projectId: string): SyncControl {
    let control = syncByProject.get(projectId);
    if (!control) {
      control = {
        cancelRequested: false,
        state: idleSyncState(projectId),
      };
      syncByProject.set(projectId, control);
    }
    return control;
  }

  async function listProjectConnections(
    projectId: string,
  ): Promise<PluginConnectionRecord[]> {
    return pluginPersistence.connections.listByProjectId(projectId);
  }

  async function listDiscoveredForProject(
    projectId: string,
  ): Promise<DiscoveredResourceRecord[]> {
    const connections = await listProjectConnections(projectId);
    const resources: DiscoveredResourceRecord[] = [];
    for (const connection of connections) {
      const items = await pluginPersistence.discoveredResources.listByConnectionId(
        connection.id,
      );
      resources.push(...items);
    }
    return resources;
  }

  async function resolveActiveEnvironments(
    projectId: string,
  ): Promise<Environment[]> {
    return environmentService.list(projectId, { includeArchived: false });
  }

  return {
    ensureProjectSeeded(projectId: string): Promise<void> {
      const inflight = seedByProject.get(projectId);
      if (inflight) {
        return inflight;
      }

      const promise = seedProjectEnvironments(
        {
          installedPlugins: pluginPersistence.installedPlugins,
          connections: pluginPersistence.connections,
          permissionGrants: pluginPersistence.permissionGrants,
          discoveredResources: pluginPersistence.discoveredResources,
          environmentService,
          configurationService,
          desiredStateService,
          bindingService,
          mappingService,
        },
        projectId,
      )
        .then(async () => {
          const environments = await environmentService.list(projectId, {
            includeArchived: true,
          });
          const connections = await listProjectConnections(projectId);
          const environmentIdsByName: Record<string, string> = {};
          for (const environment of environments) {
            environmentIdsByName[environment.name] = environment.id;
          }
          const connectionIdsByPluginId: Record<string, string> = {};
          for (const connection of connections) {
            connectionIdsByPluginId[connection.pluginId] = connection.id;
          }
          if (findingsGateway.seedForProjectContext) {
            await findingsGateway.seedForProjectContext(projectId, {
              environmentIdsByName,
              connectionIdsByPluginId,
            });
          } else {
            await findingsGateway.ensureProjectSeeded(projectId);
          }
        })
        .then(() => undefined)
        .catch((error: unknown) => {
          seedByProject.delete(projectId);
          throw error;
        });

      seedByProject.set(projectId, promise);
      return promise;
    },

    listEnvironments(projectId, options) {
      return environmentService.list(projectId, options);
    },

    createEnvironment(input: CreateEnvironmentGatewayInput) {
      return environmentService.create(input);
    },

    updateEnvironment(id: string, input: UpdateEnvironmentGatewayInput) {
      return environmentService.update(id, input);
    },

    archiveEnvironment(id: string) {
      return environmentService.archive(id);
    },

    getEnvironment(id: string) {
      return environmentService.getById(id);
    },

    listConfigurationKeys(projectId: string) {
      return configurationService.listKeys(projectId);
    },

    updateConfigurationKey(
      id: string,
      metadata: UpdateConfigurationKeyGatewayInput,
    ) {
      return configurationService.updateKeyMetadata(id, metadata);
    },

    listOccurrences(projectId: string) {
      return configurationService.listOccurrencesByProject(projectId);
    },

    listDesiredValues(projectId: string) {
      return desiredStateService.listByProject(projectId);
    },

    listDesiredValuesByEnvironment(environmentId: string) {
      return desiredStateService.listByEnvironment(environmentId);
    },

    listAppliedByEnvironment(environmentId: string) {
      return desiredStateService.listAppliedByEnvironment(environmentId);
    },

    async saveDesiredValues(
      inputs: SaveDesiredValueGatewayInput[],
    ): Promise<DesiredConfigurationValue[]> {
      const saved: DesiredConfigurationValue[] = [];
      for (const input of inputs) {
        const result = await desiredStateService.saveDesired({
          ...input,
          updatedBy: DESKTOP_CONFIG_ACTOR,
        });
        saved.push(result);
      }
      return saved;
    },

    async getEnvironmentConfigurationStatus(
      projectId: string,
      environmentId: string,
      options?: {
        drafts?: Array<{
          configurationKeyId: string;
          draftValue?: string;
          draftSecretValueRef?: string;
          draftFingerprint?: string;
          dirty: boolean;
        }>;
      },
    ) {
      const [keys, desired, occurrences, applied] = await Promise.all([
        configurationService.listKeys(projectId),
        desiredStateService.listByEnvironment(environmentId),
        configurationService.listOccurrencesByEnvironment(environmentId),
        desiredStateService.listAppliedByEnvironment(environmentId),
      ]);
      return deriveEnvironmentStatus({
        environmentId,
        keys,
        desired,
        occurrences,
        applied,
        drafts: options?.drafts,
      });
    },

    async buildApplyPlan(
      projectId: string,
      environmentId: string,
    ): Promise<ConfigurationApplyPlan> {
      const [keys, desired, occurrences, bindings, resources] = await Promise.all([
        configurationService.listKeys(projectId),
        desiredStateService.listByEnvironment(environmentId),
        configurationService.listOccurrencesByEnvironment(environmentId),
        bindingService.listByProjectId(projectId),
        listDiscoveredForProject(projectId),
      ]);
      const keyById = new Map(keys.map((key) => [key.id, key]));
      const resourceById = new Map(resources.map((resource) => [resource.id, resource]));
      const envBindings = bindings.filter(
        (binding) =>
          binding.environmentId === environmentId &&
          binding.bindingStatus === "active",
      );

      const applied = await desiredStateService.listAppliedByEnvironment(
        environmentId,
      );
      const envStatus = deriveEnvironmentStatus({
        environmentId,
        keys,
        desired,
        occurrences,
        applied,
      });
      const actionableStatuses = new Set([
        "local_changes",
        "mismatched",
        "missing_remote",
        "partially_applied",
        "remote_changed",
      ]);

      const operations = [];
      for (const value of desired) {
        const key = keyById.get(value.configurationKeyId);
        if (!key) {
          continue;
        }
        const keyStatus = envStatus.keyStatuses.find(
          (item) => item.configurationKeyId === value.configurationKeyId,
        );
        if (!keyStatus || !actionableStatuses.has(keyStatus.syncStatus)) {
          continue;
        }

        const targets = occurrences.filter(
          (occurrence) =>
            occurrence.configurationKeyId === value.configurationKeyId &&
            occurrence.resourceBindingId,
        );
        const bindingIdsFromOccurrences = [
          ...new Set(
            targets
              .map((item) => item.resourceBindingId)
              .filter((id): id is string => Boolean(id)),
          ),
        ];

        // Prefer out-of-sync resource targets; fall back to all bound occurrences
        // for the key; never fan out to every environment binding.
        const outOfSyncBindings = (keyStatus.resourceStatuses ?? [])
          .filter((item) => actionableStatuses.has(item.syncStatus))
          .map((item) => item.resourceBindingId);
        const bindingIds =
          outOfSyncBindings.length > 0
            ? outOfSyncBindings
            : bindingIdsFromOccurrences;

        for (const resourceBindingId of bindingIds) {
          const binding = envBindings.find((item) => item.id === resourceBindingId);
          const resource = binding
            ? resourceById.get(binding.discoveredResourceId)
            : undefined;
          const pluginId = binding?.pluginId ?? "unknown";
          const sensitive = key.sensitive || key.valueType === "secret";
          operations.push({
            id: crypto.randomUUID(),
            configurationKeyId: key.id,
            configurationKeyName: key.name,
            environmentId,
            resourceBindingId,
            pluginId,
            resourceName: resource?.name,
            displayValue: sensitive
              ? "••••••••"
              : (value.desiredValue ?? "(empty)"),
            sensitive,
            desiredRevision: value.revision,
            warnings: APPLY_EFFECT_WARNINGS[pluginId] ?? [],
          });
        }
      }

      const plan: ConfigurationApplyPlan = {
        id: crypto.randomUUID(),
        projectId: projectId as ConfigurationApplyPlan["projectId"],
        environmentId: environmentId as ConfigurationApplyPlan["environmentId"],
        status: "awaiting_approval",
        summary: `Apply ${operations.length} configuration change(s) to integrations`,
        operations,
        createdAt: new Date().toISOString(),
      };
      applyPlans.set(plan.id, plan);
      return plan;
    },

    async approveApplyPlan(planId: string): Promise<ConfigurationApplyResult> {
      const plan = applyPlans.get(planId);
      if (!plan) {
        throw new Error(`Apply plan not found: ${planId}`);
      }
      if (plan.status !== "awaiting_approval") {
        throw new Error(`Apply plan is not awaiting approval: ${plan.status}`);
      }

      const items = [];
      let failedCount = 0;
      const failOne =
        simulatePartialApplyOnce &&
        plan.operations.length > 1 &&
        plan.operations.some((op) => op.pluginId === "sentry");

      for (const [index, operation] of plan.operations.entries()) {
        const shouldFail =
          failOne &&
          operation.pluginId === "sentry" &&
          simulatePartialApplyOnce;
        if (shouldFail) {
          simulatePartialApplyOnce = false;
        }

        if (shouldFail) {
          const applied = await desiredStateService.recordApplied({
            configurationKeyId: operation.configurationKeyId,
            environmentId: plan.environmentId,
            projectId: plan.projectId,
            resourceBindingId: operation.resourceBindingId,
            desiredRevision: operation.desiredRevision,
            appliedFingerprint: operation.sensitive
              ? undefined
              : `fp:apply-${operation.configurationKeyId}`,
            applyExecutionId: `stub-exec-${operation.id}`,
            status: "failed",
          });
          failedCount += 1;
          items.push({
            operationId: operation.id,
            configurationKeyId: operation.configurationKeyId,
            resourceBindingId: operation.resourceBindingId,
            status: "failed" as const,
            applied,
            errorMessage: "Simulated apply failure (fixture stub)",
          });
          continue;
        }

        const desired = await desiredStateService.getDesired(
          operation.configurationKeyId,
          plan.environmentId,
        );
        const applied = await desiredStateService.recordApplied({
          configurationKeyId: operation.configurationKeyId,
          environmentId: plan.environmentId,
          projectId: plan.projectId,
          resourceBindingId: operation.resourceBindingId,
          desiredRevision: operation.desiredRevision,
          appliedFingerprint:
            desired?.valueFingerprint ??
            (operation.sensitive
              ? undefined
              : `fp:apply-${operation.configurationKeyId}-${index}`),
          applyExecutionId: `stub-exec-${operation.id}`,
          status: "applied",
        });
        items.push({
          operationId: operation.id,
          configurationKeyId: operation.configurationKeyId,
          resourceBindingId: operation.resourceBindingId,
          status: "applied" as const,
          applied,
        });
      }

      const status =
        failedCount === 0
          ? "completed"
          : failedCount === items.length
            ? "failed"
            : "partial";
      plan.status = status === "completed" ? "completed" : status;
      applyPlans.set(plan.id, plan);

      return {
        planId: plan.id,
        environmentId: plan.environmentId,
        status,
        items,
        finishedAt: new Date().toISOString(),
      };
    },

    async adoptDiscoveredKey(
      input: AdoptDiscoveredKeyGatewayInput,
    ): Promise<DesiredConfigurationValue> {
      const key = await configurationService.getKey(input.configurationKeyId);
      if (!key) {
        throw new Error(`Configuration key not found: ${input.configurationKeyId}`);
      }
      const occurrences =
        await configurationService.listOccurrencesByEnvironment(
          input.environmentId,
        );
      const relevant = occurrences.filter(
        (occurrence) => occurrence.configurationKeyId === input.configurationKeyId,
      );
      const readable = relevant.find(
        (occurrence) =>
          occurrence.valueAccess === "readable" && !key.sensitive,
      );

      const existing = await desiredStateService.getDesired(
        input.configurationKeyId,
        input.environmentId,
      );

      return desiredStateService.saveDesired({
        configurationKeyId: input.configurationKeyId,
        environmentId: input.environmentId,
        projectId: input.projectId,
        desiredValue:
          input.copyReadableValue && readable
            ? readable.observedValue
            : existing && !key.sensitive
              ? existing.desiredValue
              : undefined,
        secretValueRef: key.sensitive
          ? (relevant.find((item) => item.secretValueRef)?.secretValueRef ??
            existing?.secretValueRef)
          : undefined,
        valueFingerprint:
          relevant.find((item) => item.valueFingerprint)?.valueFingerprint ??
          existing?.valueFingerprint,
        expectedRevision: existing?.revision,
        updatedBy: DESKTOP_CONFIG_ACTOR,
      });
    },

    listDiscoveredResources(projectId: string) {
      return listDiscoveredForProject(projectId);
    },

    listBindings(projectId: string) {
      return bindingService.listByProjectId(projectId);
    },

    async attachResource(input: AttachResourceInput): Promise<ResourceBindingRecord> {
      const environment = await environmentService.getById(input.environmentId);
      if (!environment || environment.projectId !== input.projectId) {
        throw new Error("Environment does not belong to project");
      }
      return bindingService.bind({
        projectId: input.projectId,
        environmentId: input.environmentId,
        expectedProjectIdForEnvironment: input.projectId,
        discoveredResourceId: input.discoveredResourceId,
        createdBy: DESKTOP_USER_ACTOR,
      });
    },

    async moveResource(input: MoveResourceInput): Promise<ResourceBindingRecord> {
      const existing = await pluginPersistence.resourceBindings.getById(
        input.bindingId,
      );
      if (!existing) {
        throw new Error(`Resource binding not found: ${input.bindingId}`);
      }
      const environment = await environmentService.getById(input.environmentId);
      if (!environment || environment.projectId !== existing.projectId) {
        throw new Error("Environment does not belong to binding project");
      }

      const updated: ResourceBindingRecord = {
        ...existing,
        environmentId: input.environmentId,
        bindingStatus: "active",
        updatedAt: new Date().toISOString(),
      };
      await pluginPersistence.resourceBindings.save(updated);
      return updated;
    },

    detachResource(bindingId: string) {
      return bindingService.detach(bindingId);
    },

    listPendingSuggestions(projectId: string) {
      return mappingService.listPending(projectId);
    },

    async acceptSuggestion(
      input: AcceptSuggestionInput,
    ): Promise<AcceptSuggestionResult> {
      const pending = await pluginPersistence.mappingSuggestions.getById(
        input.suggestionId,
      );
      if (!pending || pending.status !== "pending") {
        throw new Error(`Pending mapping suggestion not found: ${input.suggestionId}`);
      }

      let environment: Environment | undefined;
      let environmentId = input.environmentId ?? pending.suggestedEnvironmentId;

      if (input.createEnvironment) {
        environment = await environmentService.create({
          projectId: pending.projectId,
          name: input.createEnvironment.name,
          kind: input.createEnvironment.kind,
          description: input.createEnvironment.description,
          status: "local_only",
        });
        environmentId = environment.id;
      }

      if (!environmentId) {
        throw new Error("Accepting a suggestion requires an environmentId or createEnvironment");
      }

      if (!environment) {
        const existing = await environmentService.getById(environmentId);
        if (!existing) {
          throw new Error(`Environment not found: ${environmentId}`);
        }
        environment = existing;
      }

      if (environment.projectId !== pending.projectId) {
        throw new Error("Environment does not belong to suggestion project");
      }

      const suggestion = await mappingService.accept({
        suggestionId: input.suggestionId,
        resolvedBy: DESKTOP_USER_ACTOR,
      });

      const binding = await bindingService.bind({
        projectId: pending.projectId,
        environmentId: environment.id,
        expectedProjectIdForEnvironment: environment.projectId,
        discoveredResourceId: pending.discoveredResourceId,
        createdBy: DESKTOP_USER_ACTOR,
      });

      return { suggestion, binding, environment };
    },

    rejectSuggestion(suggestionId: string) {
      return mappingService.reject({
        suggestionId,
        resolvedBy: DESKTOP_USER_ACTOR,
      });
    },

    async chooseSuggestionEnvironment(
      suggestionId: string,
      environmentId: string,
    ): Promise<EnvironmentMappingSuggestionRecord> {
      const existing = await pluginPersistence.mappingSuggestions.getById(
        suggestionId,
      );
      if (!existing || existing.status !== "pending") {
        throw new Error(`Pending mapping suggestion not found: ${suggestionId}`);
      }
      const environment = await environmentService.getById(environmentId);
      if (!environment || environment.projectId !== existing.projectId) {
        throw new Error("Environment does not belong to suggestion project");
      }
      const updated: EnvironmentMappingSuggestionRecord = {
        ...existing,
        suggestedEnvironmentId: environmentId,
        suggestedEnvironmentName: environment.name,
      };
      await pluginPersistence.mappingSuggestions.save(updated);
      return updated;
    },

    async getMatrix(projectId: string) {
      const [environments, keys, occurrences] = await Promise.all([
        resolveActiveEnvironments(projectId),
        configurationService.listKeys(projectId),
        configurationService.listOccurrencesByProject(projectId),
      ]);
      return buildConfigurationMatrix({
        projectId,
        environments,
        keys,
        occurrences,
      });
    },

    async listOpenFindings(projectId: string) {
      await findingsGateway.ensureProjectSeeded(projectId);
      return findingsGateway.listFindings({
        projectId,
        statuses: ["open", "acknowledged"],
        includeResolved: false,
      });
    },

    async getFindingsSummary(projectId: string) {
      await findingsGateway.ensureProjectSeeded(projectId);
      return findingsGateway.getProjectSummary(projectId);
    },

    async getEnvironmentFindingsSummary(
      projectId: string,
      environmentId: string,
    ) {
      await findingsGateway.ensureProjectSeeded(projectId);
      return findingsGateway.getEnvironmentSummary(projectId, environmentId);
    },

    async syncWithIntegrations(
      projectId: string,
      options?: { environmentId?: string },
    ): Promise<EnvironmentSyncResult> {
      const control = getSyncControl(projectId);
      if (control.state.inProgress) {
        throw new Error("A sync is already in progress for this project");
      }

      control.cancelRequested = false;
      const startedAt = new Date().toISOString();
      const pluginResults: EnvironmentSyncPluginResult[] = [];
      let suggestionsCreated = 0;
      let resourcesTouched = 0;
      let cancelled = false;

      const setPhase = (phase: EnvironmentSyncPhase, label?: string) => {
        control.state = {
          projectId,
          inProgress: phase !== "complete" && phase !== "cancelled" && phase !== "failed",
          phase,
          cancelled: control.cancelRequested,
          progressLabel: label,
          environmentId: options?.environmentId,
          lastResult: control.state.lastResult,
        };
      };

      setPhase("starting", "Starting read-only discovery…");
      setPhase("discovering", "Discovering resources from integrations…");

      const connections = await listProjectConnections(projectId);
      const environments = await resolveActiveEnvironments(projectId);

      for (const connection of connections) {
        if (control.cancelRequested) {
          cancelled = true;
          break;
        }

        // Simulated partial failure: Sentry fails once, then succeeds.
        if (connection.pluginId === "sentry" && sentryFailOnce) {
          sentryFailOnce = false;
          pluginResults.push({
            pluginId: connection.pluginId,
            connectionId: connection.id,
            connectionName: connection.name,
            status: "failed",
            message: "Simulated read-only discovery failure (fixture)",
          });
          continue;
        }

        try {
          const existing = await discoveryService.listByConnectionId(connection.id);
          const items = existing.map((resource) => ({
            providerResourceId: resource.providerResourceId,
            resourceType: resource.resourceType,
            name: resource.name,
            parentProviderResourceId: resource.parentProviderResourceId,
            metadata: { ...resource.metadata, lastSyncReadOnly: true },
            pluginVersion: resource.pluginVersion,
            schemaVersion: resource.schemaVersion,
          }));

          // Re-sync existing discovered items (read-only refresh). If none yet,
          // invent a lightweight discovered resource for demo sync coverage.
          const syncItems =
            items.length > 0
              ? items
              : [
                  {
                    providerResourceId: `sync:${connection.pluginId}`,
                    resourceType: "environment",
                    name: `${connection.name} (discovered)`,
                    metadata: { fixtureSync: true },
                    pluginVersion: "0.0.0",
                    schemaVersion: "1",
                  },
                ];

          const discovered = await discoveryService.sync({
            connectionId: connection.id,
            installedPluginId: connection.installedPluginId,
            pluginId: connection.pluginId,
            items: syncItems,
          });
          resourcesTouched += discovered.length;

          let createdForPlugin = 0;
          // Generate suggestions for unbound resources — never auto-accept.
          const bindings = await bindingService.listByProjectId(projectId);
          const activeBound = new Set(
            bindings
              .filter((binding) => binding.bindingStatus === "active")
              .map((binding) => binding.discoveredResourceId),
          );
          const pending = await mappingService.listPending(projectId);
          const pendingResources = new Set(
            pending.map((suggestion) => suggestion.discoveredResourceId),
          );

          for (const resource of discovered) {
            if (activeBound.has(resource.id) || pendingResources.has(resource.id)) {
              continue;
            }
            const match = environments.find((environment) =>
              resource.name.toLowerCase().includes(environment.name.toLowerCase()),
            );
            await mappingService.createSuggestion({
              projectId,
              connectionId: connection.id,
              discoveredResourceId: resource.id,
              suggestedEnvironmentId: match?.id,
              suggestedEnvironmentName: match?.name,
              confidence: match ? 0.7 : 0.4,
              reasons: match
                ? [`Name similarity with ${match.name}`]
                : ["Unmapped discovered resource"],
            });
            createdForPlugin += 1;
            suggestionsCreated += 1;
          }

          pluginResults.push({
            pluginId: connection.pluginId,
            connectionId: connection.id,
            connectionName: connection.name,
            status: "success",
            resourcesDiscovered: discovered.length,
            suggestionsCreated: createdForPlugin,
          });
        } catch (error) {
          pluginResults.push({
            pluginId: connection.pluginId,
            connectionId: connection.id,
            connectionName: connection.name,
            status: "failed",
            message: error instanceof Error ? error.message : "Discovery failed",
          });
        }
      }

      if (!cancelled) {
        setPhase("mapping", "Preparing mapping suggestions…");
        setPhase("building_matrix", "Refreshing configuration matrix…");
      }

      const finishedAt = new Date().toISOString();
      const anyFailure = pluginResults.some((result) => result.status === "failed");
      const phase: EnvironmentSyncPhase = cancelled
        ? "cancelled"
        : anyFailure
          ? "failed"
          : "complete";

      // Flip environment statuses lightly based on outcome (never provider writes).
      if (!cancelled && options?.environmentId) {
        await environmentService.update(options.environmentId, {
          status: anyFailure ? "attention_required" : "healthy",
        });
      } else if (!cancelled) {
        for (const environment of environments) {
          if (environment.status === "local_only" || environment.status === "archived") {
            continue;
          }
          if (environment.kind === "custom") {
            continue;
          }
          await environmentService.update(environment.id, {
            status: anyFailure && environment.name === "Production"
              ? "attention_required"
              : environment.status === "error"
                ? "attention_required"
                : "healthy",
          });
        }
      }

      const result: EnvironmentSyncResult = {
        projectId,
        environmentId: options?.environmentId,
        phase,
        cancelled,
        startedAt,
        finishedAt,
        plugins: pluginResults,
        suggestionsCreated,
        resourcesTouched,
      };

      control.state = {
        projectId,
        inProgress: false,
        phase,
        cancelled,
        lastResult: result,
        environmentId: options?.environmentId,
        progressLabel: cancelled
          ? "Sync cancelled"
          : anyFailure
            ? "Sync finished with partial failures"
            : "Sync complete",
      };
      control.cancelRequested = false;

      return result;
    },

    async cancelSync(projectId: string): Promise<void> {
      const control = getSyncControl(projectId);
      if (!control.state.inProgress) {
        return;
      }
      control.cancelRequested = true;
      control.state = {
        ...control.state,
        cancelled: true,
        progressLabel: "Cancelling sync…",
      };
    },

    async getSyncState(projectId: string): Promise<EnvironmentSyncState> {
      return getSyncControl(projectId).state;
    },
  };
}
