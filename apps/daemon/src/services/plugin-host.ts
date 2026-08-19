import { existsSync, mkdirSync, renameSync, rmSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { randomUUID } from "node:crypto";

import type { RayvanActor } from "@rayvan/daemon-contracts";
import {
  DevelopmentMemoryCredentialStore,
  PluginConnectionService,
  PluginInstallationService,
  ResourceBindingService,
  ResourceDiscoveryService,
  ResourceStateService,
  type ChangePlanRecord,
  type CredentialStore,
  type DiscoveredResourceRecord,
  type InstalledPluginRecord,
  type PluginConnectionRecord,
  type ResourceBindingRecord,
} from "@rayvan/local-database";
import {
  PluginPackageInstallService,
  PluginSetupSessionService,
  type PluginPersistenceBundle,
} from "@rayvan/local-database/sqlite";
import {
  installPluginPackageFromPath,
  PluginPackageError,
  resolveAllowUnsignedPlugins,
  type PluginTrustStatus,
} from "@rayvan/plugin-package";
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

export interface PluginHostContext {
  installedPluginId: string;
  connection: PluginConnectionRecord;
  discovered: DiscoveredResourceRecord[];
  bindings: ResourceBindingRecord[];
}

export interface PluginHostOptions {
  dataDir: string;
  allowUnsignedPlugins?: boolean;
  trustedPublicKeys?: readonly (Buffer | Uint8Array | string)[];
  credentials?: CredentialStore;
}

/**
 * Shared daemon plugin host: SQLite-backed installs/connections, built-in
 * example-local, and externally installed OOP packages (e.g. GitHub).
 */
export class PluginHost {
  readonly installation: PluginInstallationService;
  readonly packageInstall: PluginPackageInstallService;
  readonly setupSessions: PluginSetupSessionService;
  readonly connections: PluginConnectionService;
  readonly discovery: ResourceDiscoveryService;
  readonly bindings: ResourceBindingService;
  readonly resourceState: ResourceStateService;
  private readonly credentials: CredentialStore;
  private reconciled = false;
  private readonly oopRuntimes = new Map<
    string,
    { stop: () => Promise<void> }
  >();

  constructor(
    private readonly pluginRepos: PluginPersistenceBundle,
    private readonly pluginStack: PluginStack,
    private readonly options: PluginHostOptions,
  ) {
    this.credentials =
      options.credentials ?? new DevelopmentMemoryCredentialStore();
    this.installation = new PluginInstallationService(
      pluginRepos.installedPlugins,
    );
    this.packageInstall = new PluginPackageInstallService(
      pluginRepos.installedPlugins,
    );
    this.setupSessions = new PluginSetupSessionService(
      pluginRepos.installedPlugins,
      pluginRepos.setupSessions,
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
      id: string;
      pluginId: string;
      name: string;
      version: string;
      publisher: string;
      description?: string;
      status: string;
      host: string;
      enabled: boolean;
      reason?: string;
      trustStatus?: PluginTrustStatus | "built_in";
      trustLabel?: string;
      presentation?: PluginManifest["presentation"];
      setup?: PluginManifest["setup"];
    }>
  > {
    await this.ensureReconciled();
    const installed = await this.installation.list();
    return installed.map((record) => {
      const registered = this.pluginStack.registry.get(record.pluginId);
      const source = record.source;
      const host =
        source.type === "package"
          ? (source.hostKind ?? "out_of_process")
          : "in_process";
      const trustStatus =
        source.type === "package"
          ? source.trustStatus
          : ("built_in" as const);
      const trustLabel =
        source.type === "package"
          ? source.trustLabel
          : "Built-in";
      return {
        id: record.id,
        pluginId: record.pluginId,
        name: record.manifestSnapshot.name,
        version: record.pluginVersion,
        publisher: record.manifestSnapshot.publisher,
        description: record.manifestSnapshot.description,
        status: record.enabled ? "available" : "disabled",
        host,
        enabled: record.enabled,
        trustStatus,
        trustLabel,
        presentation: record.manifestSnapshot.presentation,
        setup: record.manifestSnapshot.setup,
        reason: registered
          ? undefined
          : source.type === "package"
            ? "Installed package; runtime registered on demand"
            : "Plugin not registered in the daemon execution stack",
      };
    });
  }

  async installFromPath(packagePath: string): Promise<InstalledPluginRecord> {
    await this.ensureReconciled();
    const allowUnsigned = resolveAllowUnsignedPlugins(
      this.options.allowUnsignedPlugins,
    );

    const stagingRoot = join(
      this.options.dataDir,
      "plugins",
      "staging",
      `${Date.now()}-${randomUUID()}`,
    );

    let installedLayout;
    try {
      installedLayout = installPluginPackageFromPath({
        packagePath,
        installRoot: stagingRoot,
        trustedPublicKeys: this.options.trustedPublicKeys ?? [],
        allowUnsignedPlugins: allowUnsigned,
        enforceHostTriple: true,
      });

      const finalRoot = join(
        this.options.dataDir,
        "plugins",
        installedLayout.manifest.id,
        installedLayout.manifest.version,
        installedLayout.targetTriple ?? "unknown",
      );
      // Capture relative binary path before staging is renamed away.
      const binaryRel = relative(stagingRoot, installedLayout.binaryPath);
      mkdirSync(dirname(finalRoot), { recursive: true });
      if (existsSync(finalRoot)) {
        rmSync(finalRoot, { recursive: true, force: true });
      }
      renameSync(stagingRoot, finalRoot);

      installedLayout = {
        ...installedLayout,
        rootDir: finalRoot,
        manifestPath: join(finalRoot, "manifest.json"),
        binaryPath: join(finalRoot, binaryRel),
      };
    } catch (error) {
      if (existsSync(stagingRoot)) {
        rmSync(stagingRoot, { recursive: true, force: true });
      }
      throw mapPluginInstallError(error, allowUnsigned);
    }

    const existing = await this.installation.getByPluginId(
      installedLayout.manifest.id,
    );
    const record = await this.packageInstall.installFromPackage({
      manifest: installedLayout.manifest,
      packagePath,
      installPath: installedLayout.rootDir,
      binaryPath: installedLayout.binaryPath,
      trustStatus: installedLayout.trustStatus,
      trustLabel: installedLayout.trustLabel,
      targetTriple: installedLayout.targetTriple,
      signerFingerprint: installedLayout.signerFingerprint,
      hostKind: "out_of_process",
      replaceExisting: Boolean(existing),
    });

    // Dynamic OOP registration is owned by DaemonRuntime (runtime map).
    return record;
  }

  async uninstallPlugin(pluginId: string): Promise<InstalledPluginRecord> {
    const installed = await this.installation.getByPluginId(pluginId);
    if (!installed) {
      throw new DaemonAppError(
        "NOT_FOUND",
        `Plugin not installed: ${pluginId}`,
      );
    }
    if (pluginId === EXAMPLE_LOCAL_PLUGIN_ID) {
      throw new DaemonAppError(
        "VALIDATION_FAILED",
        "Built-in example-local cannot be uninstalled",
      );
    }
    const runtime = this.oopRuntimes.get(pluginId);
    if (runtime) {
      await runtime.stop();
      this.oopRuntimes.delete(pluginId);
    }
    return this.installation.uninstall(installed.id);
  }

  async setPluginEnabled(
    pluginId: string,
    enabled: boolean,
  ): Promise<InstalledPluginRecord> {
    const installed = await this.installation.getByPluginId(pluginId);
    if (!installed) {
      throw new DaemonAppError(
        "NOT_FOUND",
        `Plugin not installed: ${pluginId}`,
      );
    }
    await this.installation.setEnabled(installed.id, enabled);
    const next = await this.installation.getByPluginId(pluginId);
    if (!next) {
      throw new DaemonAppError("NOT_FOUND", `Plugin not installed: ${pluginId}`);
    }
    return next;
  }

  async createConnection(input: {
    pluginId: string;
    projectId: string;
    name: string;
    metadata?: Record<string, unknown>;
  }): Promise<PluginConnectionRecord> {
    await this.ensureReconciled();
    const installed = await this.installation.getByPluginId(input.pluginId);
    if (!installed?.enabled) {
      throw new DaemonAppError(
        "PLUGIN_UNAVAILABLE",
        `Plugin ${input.pluginId} is not installed or enabled`,
        { retryable: true },
      );
    }
    return this.connections.create({
      installedPluginId: installed.id,
      projectId: input.projectId,
      name: input.name,
      status: "pending",
      metadata: input.metadata ?? {},
    });
  }

  async ensureProjectConnection(
    projectId: string,
    pluginId: string = EXAMPLE_LOCAL_PLUGIN_ID,
  ): Promise<PluginHostContext> {
    await this.ensureReconciled();
    const installed = await this.installation.getByPluginId(pluginId);
    if (!installed?.enabled) {
      throw new DaemonAppError(
        "PLUGIN_UNAVAILABLE",
        `${pluginId} plugin is not installed or enabled in the daemon host`,
        { retryable: true },
      );
    }

    const existing = (
      await this.connections.listByProjectId(projectId)
    ).find((connection) => connection.pluginId === pluginId);

    let connection = existing;
    if (!connection) {
      connection = await this.connections.create({
        installedPluginId: installed.id,
        projectId,
        name:
          pluginId === EXAMPLE_LOCAL_PLUGIN_ID
            ? EXAMPLE_LOCAL_CONNECTION_NAME
            : installed.manifestSnapshot.name,
        status: "connected",
        metadata:
          pluginId === EXAMPLE_LOCAL_PLUGIN_ID
            ? { fixture: true, host: "in_process" }
            : { host: "out_of_process" },
      });
    } else if (
      connection.status === "disconnected" ||
      connection.status === "revoked" ||
      connection.status !== "connected"
    ) {
      connection = await this.connections.markConnected(connection.id);
    }

    return this.syncConnection(connection, toPluginActorFromSystem());
  }

  async syncConnection(
    connection: PluginConnectionRecord,
    actor: PluginExecutionActor,
  ): Promise<PluginHostContext> {
    const installed = await this.installation.getByPluginId(connection.pluginId);
    if (!installed) {
      throw new DaemonAppError(
        "PLUGIN_UNAVAILABLE",
        `Installed plugin record missing for ${connection.pluginId}`,
        { retryable: true },
      );
    }

    const credentials = await this.resolveCredentials(connection);
    const discoverResult = await this.pluginStack.executionService.discover({
      pluginId: connection.pluginId,
      projectId: connection.projectId,
      actor,
      context: {
        pluginId: connection.pluginId,
        integrationId: connection.id,
        projectId: connection.projectId,
        credentials,
        connectionMetadata: connection.metadata,
      },
    });

    if (discoverResult.status !== "succeeded") {
      throw new DaemonAppError(
        "PROVIDER_OPERATION_FAILED",
        discoverResult.error?.message ?? `${connection.pluginId} discover failed`,
        { retryable: discoverResult.error?.retryable ?? true },
      );
    }

    const pluginVersion =
      this.pluginStack.registry.get(connection.pluginId)?.manifest.version ??
      installed.pluginVersion;

    const discovered = await this.discovery.sync({
      connectionId: connection.id,
      installedPluginId: installed.id,
      pluginId: connection.pluginId,
      items: discoverResult.data.map((item) => ({
        providerResourceId: item.providerResourceId,
        resourceType: item.resourceType,
        name: item.name,
        parentProviderResourceId:
          typeof item.metadata.parentProviderResourceId === "string"
            ? item.metadata.parentProviderResourceId
            : undefined,
        metadata: item.metadata,
        pluginVersion,
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
      // Auto-bind only for built-in example-local; GitHub requires explicit bind.
      if (connection.pluginId !== EXAMPLE_LOCAL_PLUGIN_ID) {
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

  async requireBinding(resourceBindingId: string): Promise<{
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
    const connection = await this.connections.getById(binding.connectionId);
    const credentials = connection
      ? await this.resolveCredentials(connection)
      : undefined;
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
        credentials,
        connectionMetadata: connection?.metadata,
      },
    });
    if (result.status !== "succeeded") {
      throw new DaemonAppError(
        "PROVIDER_OPERATION_FAILED",
        result.error?.message ?? "Resource inspection failed",
        { retryable: result.error?.retryable ?? true },
      );
    }
    const pluginVersion =
      this.pluginStack.registry.get(binding.pluginId)?.manifest.version ??
      exampleLocalPlugin.manifest.version;
    await this.resourceState.recordObserved({
      discoveredResourceId: discovered.id,
      pluginId: binding.pluginId,
      connectionId: binding.connectionId,
      state: {
        status: result.data.status,
        attributes: result.data.attributes,
        checks: result.data.checks ?? [],
      },
      pluginVersion,
      schemaVersion: discovered.schemaVersion || LOCAL_SERVICE_SCHEMA_VERSION,
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
    const connection = await this.connections.getById(binding.connectionId);
    const credentials = connection
      ? await this.resolveCredentials(connection)
      : undefined;
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
        credentials,
        connectionMetadata: connection?.metadata,
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
        credentials,
        connectionMetadata: connection?.metadata,
      },
    });
    if (planResult.status !== "succeeded") {
      throw new DaemonAppError(
        "PROVIDER_OPERATION_FAILED",
        planResult.error?.message ?? "Plan generation failed",
        { retryable: planResult.error?.retryable ?? true },
      );
    }

    const pluginVersion =
      this.pluginStack.registry.get(binding.pluginId)?.manifest.version ??
      exampleLocalPlugin.manifest.version;
    await this.resourceState.recordObserved({
      discoveredResourceId: discovered.id,
      pluginId: binding.pluginId,
      connectionId: binding.connectionId,
      state: {
        status: inspectResult.data.status,
        attributes: inspectResult.data.attributes,
        checks: inspectResult.data.checks ?? [],
      },
      pluginVersion,
      schemaVersion: discovered.schemaVersion || LOCAL_SERVICE_SCHEMA_VERSION,
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
    const connection = await this.connections.getById(binding.connectionId);
    const credentials = connection
      ? await this.resolveCredentials(connection)
      : undefined;
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
        credentials,
        connectionMetadata: connection?.metadata,
      },
    });
  }

  async verifyPlan(
    plan: ChangePlanRecord,
    approvedPlan: ApprovedChangePlan,
    applyResult: ApplyResult,
    actor: PluginExecutionActor,
  ) {
    const { sdkBinding, binding } = await this.requireBinding(
      plan.resourceBindingId,
    );
    const connection = await this.connections.getById(binding.connectionId);
    const credentials = connection
      ? await this.resolveCredentials(connection)
      : undefined;
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
        credentials,
        connectionMetadata: connection?.metadata,
      },
    });
  }

  registerOopRuntime(
    pluginId: string,
    runtime: { stop: () => Promise<void> },
  ): void {
    this.oopRuntimes.set(pluginId, runtime);
  }

  async storeConnectionAccessToken(input: {
    pluginId: string;
    connectionId: string;
    credentialType: string;
    accessToken: string;
  }): Promise<void> {
    const reference = await this.credentials.put({
      pluginId: input.pluginId,
      connectionId: input.connectionId,
      provider: "development_memory",
      credentialType: input.credentialType,
      secret: { accessToken: input.accessToken },
    });
    await this.connections.attachCredentialReference(
      input.connectionId,
      reference,
    );
    await this.connections.markConnected(input.connectionId);
  }

  private async resolveCredentials(
    connection: PluginConnectionRecord,
  ): Promise<{ accessToken?: string } | undefined> {
    if (!connection.credentialReferenceId) {
      return undefined;
    }
    const reference = await this.pluginRepos.credentialReferences.getById(
      connection.credentialReferenceId,
    );
    if (!reference) return undefined;
    const secret = await this.credentials.get(reference);
    if (typeof secret === "string") {
      return { accessToken: secret };
    }
    if (
      secret &&
      typeof secret === "object" &&
      typeof (secret as { accessToken?: unknown }).accessToken === "string"
    ) {
      return { accessToken: (secret as { accessToken: string }).accessToken };
    }
    return undefined;
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

function mapPluginInstallError(
  error: unknown,
  allowUnsigned: boolean,
): DaemonAppError {
  if (error instanceof DaemonAppError) {
    return error;
  }
  if (error instanceof PluginPackageError) {
    const hint =
      error.code === "SIGNATURE_REJECTED" && !allowUnsigned
        ? " Set RAYVAN_ALLOW_UNSIGNED_PLUGINS=1 to install unsigned development packages."
        : "";
    return new DaemonAppError(
      "VALIDATION_FAILED",
      `${error.message}${hint}`,
      {
        details: { packageErrorCode: error.code },
        cause: error,
      },
    );
  }
  if (error instanceof Error) {
    return new DaemonAppError("INTERNAL_ERROR", error.message, {
      cause: error,
    });
  }
  return new DaemonAppError(
    "INTERNAL_ERROR",
    "Plugin package installation failed",
  );
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

/** @deprecated Prefer PluginHost — retained for import compatibility. */
export { PluginHost as ExampleLocalHost };
export type { PluginHostContext as ExampleLocalHostContext };
