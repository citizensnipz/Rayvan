import type { RayvanActor } from "@rayvan/daemon-contracts";
import {
  DevelopmentMemoryCredentialStore,
  PluginConnectionService,
  PluginInstallationService,
  ResourceBindingService,
  ResourceDiscoveryService,
  ResourceStateService,
  type ChangePlanRecord,
  type DiscoveredResourceRecord,
  type InMemoryPluginPersistence,
  type PluginConnectionRecord,
  type ResourceBindingRecord,
} from "@rayvan/local-database";
import type { createPluginExecutionStack } from "@rayvan/plugin-sdk";
import type {
  ApplyResult,
  ApprovedChangePlan,
  ChangePlan,
  DesiredResourceState,
  ObservedResourceState,
  PluginExecutionActor,
  PluginManifest,
  ResourceBinding,
} from "@rayvan/plugin-sdk";
import {
  EXAMPLE_LOCAL_PLUGIN_ID,
  LOCAL_SERVICE_RESOURCE_TYPE,
  manifest as exampleLocalManifest,
  plugin as exampleLocalPlugin,
} from "@rayvan/plugin-example-local";

import { DaemonAppError } from "../errors.js";

export const EXAMPLE_LOCAL_CONNECTION_NAME = "Example Local (built-in)";
const LOCAL_SERVICE_SCHEMA_VERSION = "1.0.0";

type PluginStack = ReturnType<typeof createPluginExecutionStack>;

export interface ExampleLocalHostContext {
  installedPluginId: string;
  connection: PluginConnectionRecord;
  discovered: DiscoveredResourceRecord[];
  bindings: ResourceBindingRecord[];
}

/**
 * Owns the transitional in-process example-local plugin stack for the daemon.
 * Long-term ownership moves to crates/plugin-host (out-of-process).
 */
export class ExampleLocalHost {
  readonly installation: PluginInstallationService;
  readonly connections: PluginConnectionService;
  readonly discovery: ResourceDiscoveryService;
  readonly bindings: ResourceBindingService;
  readonly resourceState: ResourceStateService;
  private readonly credentials = new DevelopmentMemoryCredentialStore();
  private reconciled = false;

  constructor(
    private readonly pluginRepos: InMemoryPluginPersistence,
    private readonly pluginStack: PluginStack,
  ) {
    this.installation = new PluginInstallationService(
      pluginRepos.installedPlugins,
    );
    this.connections = new PluginConnectionService(
      pluginRepos.installedPlugins,
      pluginRepos.connections,
      pluginRepos.credentialReferences,
      pluginRepos.permissionGrants,
      pluginRepos.resourceBindings,
      this.credentials,
    );
    this.discovery = new ResourceDiscoveryService(
      pluginRepos.connections,
      pluginRepos.discoveredResources,
    );
    this.bindings = new ResourceBindingService(
      pluginRepos.discoveredResources,
      pluginRepos.resourceBindings,
    );
    this.resourceState = new ResourceStateService(
      pluginRepos.observedState,
      pluginRepos.desiredState,
    );
  }

  async ensureReconciled(): Promise<void> {
    if (this.reconciled) return;
    await this.installation.reconcileBuiltIns([
      { manifest: exampleLocalManifest },
    ]);
    // Ensure built-in stays enabled for the in-process host.
    const installed = await this.installation.getByPluginId(
      EXAMPLE_LOCAL_PLUGIN_ID,
    );
    if (installed && (!installed.enabled || installed.status !== "installed")) {
      await this.pluginRepos.installedPlugins.save({
        ...installed,
        enabled: true,
        status: "installed",
        updatedAt: new Date().toISOString(),
      });
    }
    this.reconciled = true;
  }

  async listPluginStatus(): Promise<
    Array<{
      pluginId: string;
      name: string;
      version: string;
      publisher: string;
      description?: string;
      status: string;
      host: string;
      enabled: boolean;
      reason?: string;
      presentation?: PluginManifest["presentation"];
    }>
  > {
    await this.ensureReconciled();
    const installed = await this.installation.getByPluginId(
      EXAMPLE_LOCAL_PLUGIN_ID,
    );
    const registered = this.pluginStack.registry.get(EXAMPLE_LOCAL_PLUGIN_ID);
    const installedManifest = installed?.manifestSnapshot;
    const base = {
      pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
      name:
        installedManifest?.name ??
        registered?.manifest.name ??
        exampleLocalManifest.name,
      version:
        installedManifest?.version ??
        registered?.manifest.version ??
        exampleLocalManifest.version,
      publisher:
        installedManifest?.publisher ??
        registered?.manifest.publisher ??
        exampleLocalManifest.publisher,
      description:
        installedManifest?.description ??
        registered?.manifest.description ??
        exampleLocalManifest.description,
      presentation:
        installedManifest?.presentation ??
        registered?.manifest.presentation ??
        exampleLocalManifest.presentation,
    };
    if (!installed || !registered) {
      return [
        {
          ...base,
          status: "unavailable",
          host: "in_process",
          enabled: false,
          reason: "example-local is not registered in the daemon plugin stack",
        },
      ];
    }
    return [
      {
        ...base,
        status: installed.enabled ? "available" : "disabled",
        host: "in_process",
        enabled: installed.enabled,
        reason:
          "Daemon hosts the TypeScript plugin execution stack in-process until crates/plugin-host is wired",
      },
    ];
  }

  async ensureProjectConnection(
    projectId: string,
  ): Promise<ExampleLocalHostContext> {
    await this.ensureReconciled();
    const installed = await this.installation.getByPluginId(
      EXAMPLE_LOCAL_PLUGIN_ID,
    );
    if (!installed?.enabled) {
      throw new DaemonAppError(
        "PLUGIN_UNAVAILABLE",
        "example-local plugin is not installed or enabled in the daemon host",
        { retryable: true },
      );
    }

    const existing = (
      await this.connections.listByProjectId(projectId)
    ).find((connection) => connection.pluginId === EXAMPLE_LOCAL_PLUGIN_ID);

    let connection = existing;
    if (!connection) {
      connection = await this.connections.create({
        installedPluginId: installed.id,
        projectId,
        name: EXAMPLE_LOCAL_CONNECTION_NAME,
        status: "connected",
        metadata: { fixture: true, host: "in_process" },
      });
    } else if (
      connection.status === "disconnected" ||
      connection.status === "revoked"
    ) {
      connection = await this.connections.markConnected(connection.id);
    } else if (connection.status !== "connected") {
      connection = await this.connections.markConnected(connection.id);
    }

    return this.syncConnection(connection, toPluginActorFromSystem());
  }

  async syncConnection(
    connection: PluginConnectionRecord,
    actor: PluginExecutionActor,
  ): Promise<ExampleLocalHostContext> {
    const installed = await this.installation.getByPluginId(connection.pluginId);
    if (!installed) {
      throw new DaemonAppError(
        "PLUGIN_UNAVAILABLE",
        `Installed plugin record missing for ${connection.pluginId}`,
        { retryable: true },
      );
    }

    const discoverResult = await this.pluginStack.executionService.discover({
      pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
      projectId: connection.projectId,
      actor,
      context: {
        pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
        integrationId: connection.id,
        projectId: connection.projectId,
      },
    });

    if (discoverResult.status !== "succeeded") {
      throw new DaemonAppError(
        "PROVIDER_OPERATION_FAILED",
        discoverResult.error?.message ?? "example-local discover failed",
        { retryable: discoverResult.error?.retryable ?? true },
      );
    }

    const discovered = await this.discovery.sync({
      connectionId: connection.id,
      installedPluginId: installed.id,
      pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
      items: discoverResult.data.map((item) => ({
        providerResourceId: item.providerResourceId,
        resourceType: item.resourceType,
        name: item.name,
        metadata: item.metadata,
        pluginVersion: exampleLocalPlugin.manifest.version,
        schemaVersion: item.schemaVersion,
      })),
    });

    const bindings: ResourceBindingRecord[] = [];
    for (const resource of discovered) {
      const existingBindings =
        await this.pluginRepos.resourceBindings.listByDiscoveredResourceId(
          resource.id,
        );
      const active = existingBindings.find(
        (binding) =>
          binding.bindingStatus === "active" &&
          binding.projectId === connection.projectId,
      );
      if (active) {
        bindings.push(active);
        continue;
      }
      if (!connection.projectId) {
        continue;
      }
      const binding = await this.bindings.bind({
        projectId: connection.projectId,
        discoveredResourceId: resource.id,
        displayName: resource.name,
        createdBy: actor,
      });
      bindings.push(binding);
    }

    const refreshed =
      (await this.connections.getById(connection.id)) ?? connection;

    return {
      installedPluginId: installed.id,
      connection: refreshed,
      discovered,
      bindings,
    };
  }

  async requireBinding(
    resourceBindingId: string,
  ): Promise<{
    binding: ResourceBindingRecord;
    discovered: DiscoveredResourceRecord;
    sdkBinding: ResourceBinding;
  }> {
    const binding =
      await this.pluginRepos.resourceBindings.getById(resourceBindingId);
    if (!binding || binding.bindingStatus === "detached") {
      throw new DaemonAppError(
        "NOT_FOUND",
        `Resource binding not found: ${resourceBindingId}`,
      );
    }
    const discovered = await this.pluginRepos.discoveredResources.getById(
      binding.discoveredResourceId,
    );
    if (!discovered) {
      throw new DaemonAppError(
        "NOT_FOUND",
        `Discovered resource missing for binding ${resourceBindingId}`,
      );
    }
    return {
      binding,
      discovered,
      sdkBinding: toSdkBinding(binding, discovered),
    };
  }

  async inspectBinding(
    resourceBindingId: string,
    actor: PluginExecutionActor,
  ): Promise<ObservedResourceState> {
    const { binding, discovered, sdkBinding } =
      await this.requireBinding(resourceBindingId);
    const result = await this.pluginStack.executionService.inspect({
      pluginId: binding.pluginId,
      projectId: binding.projectId,
      environmentId: binding.environmentId,
      resourceId: binding.id,
      actor,
      context: {
        pluginId: binding.pluginId,
        integrationId: binding.connectionId,
        resource: sdkBinding,
      },
    });
    if (result.status !== "succeeded") {
      throw new DaemonAppError(
        "PROVIDER_OPERATION_FAILED",
        result.error?.message ?? "Resource inspection failed",
        { retryable: result.error?.retryable ?? true },
      );
    }
    await this.resourceState.recordObserved({
      discoveredResourceId: discovered.id,
      pluginId: binding.pluginId,
      connectionId: binding.connectionId,
      state: {
        status: result.data.status,
        attributes: result.data.attributes,
        checks: result.data.checks ?? [],
      },
      pluginVersion: exampleLocalPlugin.manifest.version,
      schemaVersion: LOCAL_SERVICE_SCHEMA_VERSION,
      observedAt: result.data.observedAt,
      sourceExecutionId: result.executionId,
    });
    return result.data;
  }

  async planForBinding(input: {
    resourceBindingId: string;
    desiredAttributes: Record<string, unknown>;
    actor: PluginExecutionActor;
  }): Promise<{
    binding: ResourceBindingRecord;
    discovered: DiscoveredResourceRecord;
    observed: ObservedResourceState;
    plan: ChangePlan;
    executionId: string;
  }> {
    const { binding, discovered, sdkBinding } = await this.requireBinding(
      input.resourceBindingId,
    );
    const inspectResult = await this.pluginStack.executionService.inspect({
      pluginId: binding.pluginId,
      projectId: binding.projectId,
      environmentId: binding.environmentId,
      resourceId: binding.id,
      actor: input.actor,
      context: {
        pluginId: binding.pluginId,
        integrationId: binding.connectionId,
        resource: sdkBinding,
      },
    });
    if (inspectResult.status !== "succeeded") {
      throw new DaemonAppError(
        "PROVIDER_OPERATION_FAILED",
        inspectResult.error?.message ?? "Inspect before plan failed",
        { retryable: inspectResult.error?.retryable ?? true },
      );
    }

    const desired: DesiredResourceState = {
      resourceId: binding.id,
      pluginId: binding.pluginId,
      resourceType: discovered.resourceType,
      attributes: input.desiredAttributes,
    };

    const planResult = await this.pluginStack.executionService.plan({
      pluginId: binding.pluginId,
      projectId: binding.projectId,
      environmentId: binding.environmentId,
      resourceId: binding.id,
      actor: input.actor,
      context: {
        pluginId: binding.pluginId,
        integrationId: binding.connectionId,
        resource: sdkBinding,
        observed: inspectResult.data,
        desired,
      },
    });
    if (planResult.status !== "succeeded") {
      throw new DaemonAppError(
        "PROVIDER_OPERATION_FAILED",
        planResult.error?.message ?? "Plan generation failed",
        { retryable: planResult.error?.retryable ?? true },
      );
    }

    await this.resourceState.recordObserved({
      discoveredResourceId: discovered.id,
      pluginId: binding.pluginId,
      connectionId: binding.connectionId,
      state: {
        status: inspectResult.data.status,
        attributes: inspectResult.data.attributes,
        checks: inspectResult.data.checks ?? [],
      },
      pluginVersion: exampleLocalPlugin.manifest.version,
      schemaVersion: LOCAL_SERVICE_SCHEMA_VERSION,
      observedAt: inspectResult.data.observedAt,
      sourceExecutionId: inspectResult.executionId,
    });

    return {
      binding,
      discovered,
      observed: inspectResult.data,
      plan: planResult.data,
      executionId: planResult.executionId,
    };
  }

  async applyPlan(
    plan: ChangePlanRecord,
    approvedPlan: ApprovedChangePlan,
    actor: PluginExecutionActor,
  ) {
    const { binding, sdkBinding } = await this.requireBinding(
      plan.resourceBindingId,
    );
    if (binding.pluginId !== plan.pluginId) {
      throw new DaemonAppError(
        "VALIDATION_FAILED",
        "Change plan plugin does not match resource binding",
      );
    }
    return this.pluginStack.executionService.apply({
      pluginId: plan.pluginId,
      projectId: plan.projectId,
      environmentId: plan.environmentId,
      resourceId: plan.resourceBindingId,
      actor,
      context: {
        pluginId: plan.pluginId,
        integrationId: plan.connectionId,
        resource: sdkBinding,
        approvedPlan,
      },
    });
  }

  async verifyPlan(
    plan: ChangePlanRecord,
    approvedPlan: ApprovedChangePlan,
    applyResult: ApplyResult,
    actor: PluginExecutionActor,
  ) {
    const { sdkBinding } = await this.requireBinding(plan.resourceBindingId);
    return this.pluginStack.executionService.verify({
      pluginId: plan.pluginId,
      projectId: plan.projectId,
      environmentId: plan.environmentId,
      resourceId: plan.resourceBindingId,
      actor,
      context: {
        pluginId: plan.pluginId,
        integrationId: plan.connectionId,
        resource: sdkBinding,
        approvedPlan,
        applyResult,
      },
    });
  }
}

function toSdkBinding(
  binding: ResourceBindingRecord,
  discovered: DiscoveredResourceRecord,
): ResourceBinding {
  return {
    resourceId: binding.id,
    pluginId: binding.pluginId,
    providerResourceId: discovered.providerResourceId,
    resourceType: discovered.resourceType || LOCAL_SERVICE_RESOURCE_TYPE,
    projectId: binding.projectId,
    environmentId: binding.environmentId,
  };
}

function toPluginActorFromSystem(): PluginExecutionActor {
  return { type: "system", id: "daemon" };
}

export function toPluginExecutionActor(
  actor: RayvanActor,
): PluginExecutionActor {
  if (actor.type === "mcp_client") {
    return { type: "mcp_agent", id: actor.id };
  }
  if (actor.type === "user" || actor.type === "desktop") {
    return { type: "user", id: actor.id };
  }
  return { type: "system", id: "daemon" };
}
