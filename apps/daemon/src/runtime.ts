import { randomUUID } from "node:crypto";
import { mkdirSync } from "node:fs";
import { join } from "node:path";

import {
  LocalClientCredentialStore,
  databasePath,
  defaultRayvanDataDir,
  daemonEndpointPath,
  defaultRayvanRuntimeDir,
} from "@rayvan/daemon-client";
import {
  BUILT_IN_LOCAL_CLIENT_IDS,
  BUILT_IN_PERMISSION_PROFILES,
  DAEMON_PROTOCOL_VERSION,
  DaemonMethods,
  type BuiltInPermissionProfileId,
  type DaemonHandshakeRequest,
  type DaemonHandshakeResponse,
  type DaemonStatus,
  type LocalClientRecord,
  type McpApprovalPolicy,
  type McpPermission,
  type RayvanActor,
} from "@rayvan/daemon-contracts";
import {
  ConfigurationDesiredStateService,
  ConfigurationService,
  EnvironmentService,
  FindingLifecycleService,
  ProjectService,
  ChangePlanService,
  ChangeApprovalService,
  FindingSummaryService,
  SqliteFindingRepository,
  SqliteFindingLifecycleEventRepository,
  createInMemoryPluginPersistence,
  type InMemoryPluginPersistence,
  MIGRATION_VERSION,
} from "@rayvan/local-database";
import {
  LocalDatabaseConnection,
  SqliteProjectRepository,
  SqliteEnvironmentRepository,
  SqliteConfigurationKeyRepository,
  SqliteConfigurationOccurrenceRepository,
  SqliteDesiredConfigurationValueRepository,
  SqliteAppliedConfigurationStateRepository,
} from "@rayvan/local-database/sqlite";
import { createPluginExecutionStack } from "@rayvan/plugin-sdk";
import { plugin as exampleLocalPlugin } from "@rayvan/plugin-example-local";

import type { SessionContext } from "./auth/session.js";
import {
  permissionsForProfile,
  requireEnvironmentScope,
  requirePermission,
  requireProjectScope,
} from "./auth/session.js";
import { DaemonAppError, toDaemonError } from "./errors.js";
import { ControlPlaneRepository } from "./repos/control-plane.js";
import { EventBus } from "./services/event-bus.js";
import {
  ExampleLocalHost,
  toPluginExecutionActor,
} from "./services/example-local-host.js";
import { DaemonSecretStore } from "./services/secrets.js";

const DAEMON_VERSION = "0.0.1";

export interface DaemonRuntimeOptions {
  dataDir?: string;
  runtimeDir?: string;
  endpoint?: string;
  provisionSystemClients?: boolean;
  allowUnauthenticatedTestClient?: boolean;
}

export class DaemonRuntime {
  readonly dataDir: string;
  readonly runtimeDir: string;
  readonly endpoint: string;
  readonly db: LocalDatabaseConnection;
  readonly control: ControlPlaneRepository;
  readonly events = new EventBus();
  readonly credentials: LocalClientCredentialStore;
  readonly secrets: DaemonSecretStore;
  readonly startedAt = Date.now();
  readonly projectService: ProjectService;
  readonly environmentService: EnvironmentService;
  readonly configurationService: ConfigurationService;
  readonly desiredStateService: ConfigurationDesiredStateService;
  readonly findingService: FindingLifecycleService;
  readonly findingSummary: FindingSummaryService;
  readonly changePlanService: ChangePlanService;
  readonly changeApprovalService: ChangeApprovalService;
  readonly pluginStack: ReturnType<typeof createPluginExecutionStack>;
  readonly pluginRepos: InMemoryPluginPersistence;
  readonly exampleLocalHost: ExampleLocalHost;
  private readonly connectedSessions = new Map<string, SessionContext>();
  private readonly allowUnauthenticatedTestClient: boolean;
  private shuttingDown = false;

  constructor(options: DaemonRuntimeOptions = {}) {
    this.dataDir = options.dataDir ?? defaultRayvanDataDir();
    this.runtimeDir = options.runtimeDir ?? defaultRayvanRuntimeDir();
    this.endpoint = options.endpoint ?? daemonEndpointPath(this.runtimeDir);
    this.allowUnauthenticatedTestClient =
      options.allowUnauthenticatedTestClient === true ||
      process.env.RAYVAN_ALLOW_UNAUTHENTICATED_TEST_CLIENT === "1";
    mkdirSync(this.dataDir, { recursive: true, mode: 0o700 });
    mkdirSync(this.runtimeDir, { recursive: true, mode: 0o700 });

    const dbPath = databasePath(this.dataDir);
    this.db = new LocalDatabaseConnection(dbPath);
    this.control = new ControlPlaneRepository(this.db);
    this.control.seedPermissionProfiles();
    this.credentials = new LocalClientCredentialStore(
      join(this.dataDir, "credentials", "local-clients.json"),
    );
    if (options.provisionSystemClients !== false) {
      this.provisionSystemClient(
        BUILT_IN_LOCAL_CLIENT_IDS.desktop,
        "Rayvan Desktop",
        "desktop",
      );
      this.provisionSystemClient(BUILT_IN_LOCAL_CLIENT_IDS.cli, "Rayvan CLI", "cli");
    }
    this.secrets = new DaemonSecretStore(join(this.dataDir, "secrets"));

    const projects = new SqliteProjectRepository(this.db);
    const environments = new SqliteEnvironmentRepository(this.db);
    const keys = new SqliteConfigurationKeyRepository(this.db);
    const occurrences = new SqliteConfigurationOccurrenceRepository(this.db);
    const desired = new SqliteDesiredConfigurationValueRepository(this.db);
    const applied = new SqliteAppliedConfigurationStateRepository(this.db);
    const findings = new SqliteFindingRepository(this.db);
    const lifecycle = new SqliteFindingLifecycleEventRepository(this.db);
    this.pluginRepos = createInMemoryPluginPersistence();

    this.projectService = new ProjectService(projects);
    this.environmentService = new EnvironmentService(environments);
    this.configurationService = new ConfigurationService(keys, occurrences);
    this.desiredStateService = new ConfigurationDesiredStateService(
      keys,
      desired,
      applied,
    );
    this.findingService = new FindingLifecycleService(findings, lifecycle);
    this.findingSummary = new FindingSummaryService(findings);
    this.changePlanService = new ChangePlanService(this.pluginRepos.changePlans);
    this.changeApprovalService = new ChangeApprovalService(
      this.pluginRepos.changePlans,
      this.pluginRepos.changePlanApprovals,
      this.pluginRepos.changeApplies,
      this.pluginRepos.changeVerifications,
    );

    // Transitional: daemon hosts the TS plugin stack in-process until
    // crates/plugin-host (out-of-process) is the connected runtime.
    this.pluginStack = createPluginExecutionStack({
      plugins: [exampleLocalPlugin],
    });
    this.exampleLocalHost = new ExampleLocalHost(
      this.pluginRepos,
      this.pluginStack,
    );
    void this.exampleLocalHost.ensureReconciled();
  }

  recoverIncompleteOperations(): void {
    for (const op of this.control.listIncompleteOperations()) {
      if (op.status === "running") {
        // Provider apply interrupted → mark uncertain/failed safely (no blind retry).
        if (op.type === "change_apply") {
          this.control.updateOperation(op.id, {
            status: "failed",
            finishedAt: new Date().toISOString(),
            safeError: {
              code: "PROVIDER_OPERATION_FAILED",
              message:
                "Apply interrupted by daemon restart; verification required before retry",
              retryable: true,
            },
          });
        } else {
          this.control.updateOperation(op.id, {
            status: "failed",
            finishedAt: new Date().toISOString(),
            safeError: {
              code: "INTERNAL_ERROR",
              message: "Operation interrupted by daemon restart",
              retryable: true,
            },
          });
        }
        this.control.releaseResourceLocksForOperation(op.id);
      }
      // waiting_for_approval and queued remain for resume/decision
    }
  }

  status(): DaemonStatus {
    return {
      state: this.shuttingDown ? "shutting_down" : "ready",
      version: DAEMON_VERSION,
      protocolVersion: DAEMON_PROTOCOL_VERSION,
      pid: process.pid,
      uptimeMs: Date.now() - this.startedAt,
      databasePath: databasePath(this.dataDir),
      databaseSchemaVersion: MIGRATION_VERSION,
      endpoint: this.endpoint,
      connectedClients: this.connectedSessions.size,
      activeOperations: this.control
        .listOperations({ limit: 500 })
        .filter((o) => ["queued", "running", "waiting_for_approval"].includes(o.status))
        .length,
      pendingApprovals: this.control.listApprovals({ status: "pending" }).length,
      pluginHostStatus: "ready",
    };
  }

  registerSession(session: SessionContext): void {
    this.connectedSessions.set(session.sessionId, session);
  }

  unregisterSession(sessionId: string): void {
    this.connectedSessions.delete(sessionId);
  }

  broadcast(event: Parameters<EventBus["emit"]>[0]): void {
    const emitted = this.events.emit(event);
    // Sessions that subscribed receive via IPC server wiring.
    void emitted;
  }

  async handshake(params: DaemonHandshakeRequest): Promise<DaemonHandshakeResponse> {
    if (params.protocolVersion !== DAEMON_PROTOCOL_VERSION) {
      throw new DaemonAppError(
        "DAEMON_VERSION_MISMATCH",
        `Unsupported protocol version ${params.protocolVersion}`,
        { details: { expected: DAEMON_PROTOCOL_VERSION } },
      );
    }

    let client: LocalClientRecord | undefined;
    let permissions: ReadonlySet<McpPermission>;
    if (params.clientId) {
      client = this.control.getClient(params.clientId) ?? undefined;
      if (!client) {
        throw new DaemonAppError(
          "CLIENT_NOT_REGISTERED",
          "Local client is not registered",
        );
      }
      if (client.status === "revoked" || client.status === "expired") {
        throw new DaemonAppError("CLIENT_REVOKED", "Local client is revoked");
      }
      if (
        !params.clientCredential ||
        !this.credentials.verify(params.clientId, params.clientCredential)
      ) {
        throw new DaemonAppError("UNAUTHENTICATED", "Invalid client credential");
      }
      permissions = permissionsForProfile(client.permissionProfileId);
      this.control.touchClient(client.id, true);
    } else if (params.clientType === "test" && this.allowUnauthenticatedTestClient) {
      permissions = new Set(BUILT_IN_PERMISSION_PROFILES.administrator);
    } else {
      throw new DaemonAppError(
        "UNAUTHENTICATED",
        "Local clients must provide a registered clientId and credential",
      );
    }

    const sessionId = `sess_${randomUUID()}`;
    return {
      protocolVersion: DAEMON_PROTOCOL_VERSION,
      daemonVersion: DAEMON_VERSION,
      sessionId,
      capabilities: [...permissions],
      authenticatedClientId: client?.id,
      permissionProfileId: client?.permissionProfileId,
    };
  }

  async dispatch(
    session: SessionContext,
    method: string,
    params: unknown,
  ): Promise<unknown> {
    if (session.client) {
      const current = this.control.getClient(session.client.id);
      if (!current || current.status !== "active") {
        throw new DaemonAppError("CLIENT_REVOKED", "Local client is revoked");
      }
      session.client = current;
      session.permissions = permissionsForProfile(current.permissionProfileId);
      session.projectScopes = resolveProjectScopes(current);
      session.environmentScopes = current.environmentScopes ?? "*";
    }
    const startedAt = new Date().toISOString();
    const correlationId = randomUUID();
    try {
      const result = await this.dispatchInner(session, method, params, correlationId);
      if (session.client) {
        this.control.touchClient(session.client.id, false);
        this.control.insertAudit({
          clientId: session.client.id,
          daemonMethod: method,
          startedAt,
          finishedAt: new Date().toISOString(),
          status: "succeeded",
          affectedObjectIds: [],
          safeSummary: `${method} succeeded`,
          contactedRemote: method.includes("sync") || method.includes("apply"),
          mutatedRemote: method.includes("apply"),
          correlationId,
          projectId: readString(params, "projectId"),
          environmentId: readString(params, "environmentId"),
        });
      }
      return result;
    } catch (error) {
      if (session.client) {
        const serialized = toDaemonError(error, correlationId);
        this.control.insertAudit({
          clientId: session.client.id,
          daemonMethod: method,
          startedAt,
          finishedAt: new Date().toISOString(),
          status:
            serialized.code === "PERMISSION_DENIED" ||
            serialized.code === "PROJECT_SCOPE_DENIED"
              ? "denied"
              : "failed",
          affectedObjectIds: [],
          safeSummary: `${method} failed`,
          errorCode: serialized.code,
          contactedRemote: false,
          mutatedRemote: false,
          correlationId,
        });
      }
      throw error;
    }
  }

  private async dispatchInner(
    session: SessionContext,
    method: string,
    params: unknown,
    correlationId: string,
  ): Promise<unknown> {
    const p = asRecord(params);

    switch (method) {
      case DaemonMethods.handshake:
        // Handled before session fully exists; keep for completeness.
        return this.handshake(p as unknown as DaemonHandshakeRequest);
      case DaemonMethods.ping:
        return { ok: true, ts: new Date().toISOString() };
      case DaemonMethods.status:
        requirePermission(session, "daemon:read");
        return this.status();
      case DaemonMethods.diagnostics:
        requirePermission(session, "daemon:read");
        return {
          status: this.status(),
          lockPath: join(this.runtimeDir, "rayvand.lock"),
          runtimeDir: this.runtimeDir,
          dataDir: this.dataDir,
          recentErrors: [],
          redacted: true as const,
        };
      case DaemonMethods.subscribe:
        session.subscribed = true;
        return { subscribed: true };
      case DaemonMethods.shutdown:
        requirePermission(session, "mcp_clients:manage");
        this.shuttingDown = true;
        setTimeout(() => process.emit("SIGTERM" as NodeJS.Signals), 50);
        return { shuttingDown: true };

      case DaemonMethods.listProjects: {
        requirePermission(session, "projects:read");
        const projects = await this.projectService.list({
          includeArchived: Boolean(p.includeArchived),
        });
        return filterProjects(projects, session);
      }
      case DaemonMethods.getProject: {
        requirePermission(session, "projects:read");
        const projectId = requireString(p, "projectId");
        requireProjectScope(session, projectId);
        return this.projectService.getById(projectId);
      }
      case DaemonMethods.getProjectOverview: {
        requirePermission(session, "projects:read");
        const projectId = requireString(p, "projectId");
        requireProjectScope(session, projectId);
        const project = await this.projectService.getById(projectId);
        const environments = await this.environmentService.list(projectId);
        const findings = await this.findingService.list({ projectId });
        return {
          project,
          environmentCount: environments.length,
          openFindings: findings.filter((f) => f.status === "open").length,
        };
      }
      case DaemonMethods.createProject: {
        requirePermission(session, "mcp_clients:manage");
        return this.projectService.create({
          name: requireString(p, "name"),
          description: optionalString(p, "description"),
        });
      }
      case DaemonMethods.updateProject: {
        requirePermission(session, "mcp_clients:manage");
        const projectId = requireString(p, "projectId");
        requireProjectScope(session, projectId);
        const hasName = Object.prototype.hasOwnProperty.call(p, "name");
        const hasDescription =
          Object.prototype.hasOwnProperty.call(p, "description");
        let updated =
          hasName || hasDescription
            ? await this.projectService.update(projectId, {
                name: hasName ? optionalString(p, "name") : undefined,
                description: hasDescription
                  ? optionalString(p, "description")
                  : undefined,
              })
            : await this.projectService.getById(projectId);
        if (!updated) {
          throw new DaemonAppError("NOT_FOUND", "Project not found");
        }
        if (typeof p.archived === "boolean") {
          updated = p.archived
            ? await this.projectService.archive(projectId)
            : await this.projectService.restore(projectId);
        }
        this.events.emit({
          type: "project_changed",
          projectId,
          actor: session.actor,
          correlationId,
          payload: {
            projectId,
            action: typeof p.archived === "boolean" ? "archived" : "updated",
          },
        });
        return updated;
      }

      case DaemonMethods.listEnvironments: {
        requirePermission(session, "environments:read");
        const projectId = requireString(p, "projectId");
        requireProjectScope(session, projectId);
        const environments = await this.environmentService.list(projectId, {
          includeArchived: Boolean(p.includeArchived),
        });
        return filterEnvironments(environments, session);
      }
      case DaemonMethods.getEnvironment: {
        requirePermission(session, "environments:read");
        const environmentId = requireString(p, "environmentId");
        const env = await this.environmentService.getById(environmentId);
        if (!env) return null;
        requireProjectScope(session, env.projectId);
        requireEnvironmentScope(session, env.id);
        return env;
      }
      case DaemonMethods.createEnvironment: {
        requirePermission(session, "environments:write");
        const projectId = requireString(p, "projectId");
        requireProjectScope(session, projectId);
        const created = await this.environmentService.create({
          projectId,
          name: requireString(p, "name"),
          kind: requireString(p, "kind") as never,
          slug: optionalString(p, "slug"),
          description: optionalString(p, "description"),
        });
        this.events.emit({
          type: "environment_changed",
          projectId,
          actor: session.actor,
          correlationId,
          payload: { environmentId: created.id, action: "created" },
        });
        return created;
      }
      case DaemonMethods.updateEnvironment: {
        requirePermission(session, "environments:write");
        const environmentId = requireString(p, "environmentId");
        const existing = await this.environmentService.getById(environmentId);
        if (!existing) {
          throw new DaemonAppError("NOT_FOUND", "Environment not found");
        }
        requireProjectScope(session, existing.projectId);
        requireEnvironmentScope(session, existing.id);
        const updated = await this.environmentService.update(environmentId, {
          name: optionalString(p, "name"),
          description: optionalString(p, "description"),
          kind: optionalString(p, "kind") as never,
          status: optionalString(p, "status") as never,
        });
        this.events.emit({
          type: "environment_changed",
          projectId: existing.projectId,
          actor: session.actor,
          correlationId,
          payload: { environmentId, action: "updated" },
        });
        return updated;
      }
      case DaemonMethods.archiveEnvironment: {
        requirePermission(session, "environments:write");
        const environmentId = requireString(p, "environmentId");
        const existing = await this.environmentService.getById(environmentId);
        if (!existing) {
          throw new DaemonAppError("NOT_FOUND", "Environment not found");
        }
        requireProjectScope(session, existing.projectId);
        requireEnvironmentScope(session, existing.id);
        const archived = await this.environmentService.archive(environmentId);
        this.events.emit({
          type: "environment_changed",
          projectId: existing.projectId,
          actor: session.actor,
          correlationId,
          payload: { environmentId, action: "archived" },
        });
        return archived;
      }
      case DaemonMethods.compareEnvironments: {
        requirePermission(session, "environments:read");
        const projectId = requireString(p, "projectId");
        requireProjectScope(session, projectId);
        const leftId = requireString(p, "leftEnvironmentId");
        const rightId = requireString(p, "rightEnvironmentId");
        requireEnvironmentScope(session, leftId);
        requireEnvironmentScope(session, rightId);
        const left = await this.environmentService.getById(leftId);
        const right = await this.environmentService.getById(rightId);
        return { left, right, projectId };
      }

      case DaemonMethods.listConfigurationKeys: {
        requirePermission(session, "configuration:read");
        const projectId = requireString(p, "projectId");
        requireProjectScope(session, projectId);
        return this.configurationService.listKeys(projectId);
      }
      case DaemonMethods.getConfigurationKey: {
        requirePermission(session, "configuration:read");
        const keyId = requireString(p, "configurationKeyId");
        const key = await this.configurationService.getKey(keyId);
        if (!key) return null;
        requireProjectScope(session, key.projectId);
        return sanitizeKey(key);
      }
      case DaemonMethods.findConfigurationUsage: {
        requirePermission(session, "configuration:read");
        const keyId = requireString(p, "configurationKeyId");
        const key = await this.configurationService.getKey(keyId);
        if (!key) throw new DaemonAppError("NOT_FOUND", "Configuration key not found");
        requireProjectScope(session, key.projectId);
        const occurrences = await this.configurationService.listOccurrencesByKey(keyId);
        return {
          key: sanitizeKey(key),
          occurrences: occurrences.map((occurrence) =>
            sanitizeOccurrence(occurrence, key.sensitive),
          ),
        };
      }
      case DaemonMethods.getConfigurationStatus: {
        requirePermission(session, "configuration:read");
        const projectId = requireString(p, "projectId");
        const environmentId = optionalString(p, "environmentId");
        requireProjectScope(session, projectId);
        if (environmentId) {
          requireEnvironmentScope(session, environmentId);
          await this.requireEnvironmentInProject(projectId, environmentId);
        }
        const [keys, desired, applied, occurrences] = await Promise.all([
          this.configurationService.listKeys(projectId),
          environmentId
            ? this.desiredStateService.listByEnvironment(environmentId)
            : this.desiredStateService.listByProject(projectId),
          environmentId
            ? this.desiredStateService.listAppliedByEnvironment(environmentId)
            : this.desiredStateService.listAppliedByProject(projectId),
          environmentId
            ? this.configurationService.listOccurrencesByEnvironment(environmentId)
            : this.configurationService.listOccurrencesByProject(projectId),
        ]);
        const sensitivity = new Map(keys.map((key) => [key.id, key.sensitive]));
        return {
          projectId,
          environmentId,
          keys: keys.map(sanitizeKey),
          desired: desired.map(sanitizeDesired),
          applied,
          occurrences: occurrences.map((occurrence) =>
            sanitizeOccurrence(
              occurrence,
              sensitivity.get(occurrence.configurationKeyId) ?? false,
            ),
          ),
        };
      }
      case DaemonMethods.listUnmanagedConfiguration: {
        requirePermission(session, "configuration:read");
        const projectId = requireString(p, "projectId");
        const environmentId = optionalString(p, "environmentId");
        requireProjectScope(session, projectId);
        if (environmentId) {
          requireEnvironmentScope(session, environmentId);
          await this.requireEnvironmentInProject(projectId, environmentId);
        }
        const [keys, occurrences] = await Promise.all([
          this.configurationService.listKeys(projectId),
          environmentId
            ? this.configurationService.listOccurrencesByEnvironment(environmentId)
            : this.configurationService.listOccurrencesByProject(projectId),
        ]);
        const keysById = new Map(keys.map((key) => [key.id, key]));
        return occurrences
          .filter((occurrence) => {
            const key = keysById.get(occurrence.configurationKeyId);
            if (!key || key.source !== "discovered") return false;
            return occurrence.scope !== "ignored";
          })
          .map((occurrence) =>
            sanitizeOccurrence(
              occurrence,
              keysById.get(occurrence.configurationKeyId)?.sensitive ?? false,
            ),
          );
      }
      case DaemonMethods.getEnvironmentConfiguration: {
        requirePermission(session, "configuration:read");
        const projectId = requireString(p, "projectId");
        const environmentId = requireString(p, "environmentId");
        requireProjectScope(session, projectId);
        requireEnvironmentScope(session, environmentId);
        await this.requireEnvironmentInProject(projectId, environmentId);
        const keys = await this.configurationService.listKeys(projectId);
        const values = [];
        for (const key of keys) {
          const desired = await this.desiredStateService.getDesired(
            key.id,
            environmentId,
          );
          values.push({
            key: sanitizeKey(key),
            desired: desired ? sanitizeDesired(desired) : null,
          });
        }
        return { projectId, environmentId, values };
      }
      case DaemonMethods.setConfigurationValue: {
        requirePermission(session, "configuration:write");
        return this.setConfigurationValue(session, p, correlationId, false);
      }
      case DaemonMethods.setSensitiveConfigurationValue: {
        requirePermission(session, "configuration:write");
        return this.setConfigurationValue(session, p, correlationId, true);
      }
      case DaemonMethods.clearConfigurationValue: {
        requirePermission(session, "configuration:write");
        const configurationKeyId = requireString(p, "configurationKeyId");
        const environmentId = requireString(p, "environmentId");
        const projectId = requireString(p, "projectId");
        requireProjectScope(session, projectId);
        requireEnvironmentScope(session, environmentId);
        await this.requireEnvironmentInProject(projectId, environmentId);
        await this.requireConfigurationKeyInProject(projectId, configurationKeyId);
        const expectedRevision = requireNumber(p, "expectedRevision");
        const saved = await this.desiredStateService.saveDesired({
          configurationKeyId,
          environmentId,
          projectId,
          desiredValue: undefined,
          updatedBy: actorToConfigActor(session.actor),
          expectedRevision,
        });
        this.events.emit({
          type: "configuration_changed",
          projectId,
          actor: session.actor,
          correlationId,
          payload: { configurationKeyId, environmentId, action: "cleared" },
        });
        return { saved: sanitizeDesired(saved), remoteStateAffected: false };
      }
      case DaemonMethods.setConfigurationMetadata: {
        requirePermission(session, "configuration:write");
        const projectId = requireString(p, "projectId");
        const configurationKeyId = requireString(p, "configurationKeyId");
        requireProjectScope(session, projectId);
        const key = await this.configurationService.getKey(configurationKeyId);
        if (!key || key.projectId !== projectId) {
          throw new DaemonAppError("NOT_FOUND", "Configuration key not found");
        }
        const updated = await this.configurationService.updateKeyMetadata(
          configurationKeyId,
          {
            description: optionalString(p, "description"),
            valueType: optionalString(p, "valueType") as never,
            required: typeof p.required === "boolean" ? p.required : undefined,
            sensitive: typeof p.sensitive === "boolean" ? p.sensitive : undefined,
          },
        );
        this.events.emit({
          type: "configuration_changed",
          projectId,
          actor: session.actor,
          correlationId,
          payload: { configurationKeyId, action: "metadata_updated" },
        });
        return { key: sanitizeKey(updated), remoteStateAffected: false };
      }
      case DaemonMethods.setConfigurationTargets:
        return this.setConfigurationTargets(session, p, correlationId);
      case DaemonMethods.removeConfigurationTarget:
        return this.removeConfigurationTarget(session, p, correlationId);
      case DaemonMethods.adoptDiscoveredConfiguration:
        return this.adoptDiscoveredConfiguration(session, p, correlationId);
      case DaemonMethods.ignoreDiscoveredConfiguration:
        return this.ignoreDiscoveredConfiguration(session, p, correlationId);
      case DaemonMethods.revealSensitiveConfigurationValue: {
        requirePermission(session, "configuration:read_sensitive");
        throw new DaemonAppError(
          "SECRET_ACCESS_DENIED",
          "Sensitive reveal requires desktop approval policy; use approvals queue",
        );
      }

      case DaemonMethods.listFindings: {
        requirePermission(session, "findings:read");
        const projectId = requireString(p, "projectId");
        requireProjectScope(session, projectId);
        const status = optionalString(p, "status");
        const findings = await this.findingService.list({
          projectId,
          statuses: status ? [status as never] : undefined,
          environmentId: optionalString(p, "environmentId"),
        });
        return filterEnvironmentScoped(findings, session);
      }
      case DaemonMethods.getFinding: {
        requirePermission(session, "findings:read");
        const findingId = requireString(p, "findingId");
        const finding = await this.findingService.get(findingId);
        if (!finding) return null;
        requireProjectScope(session, finding.projectId);
        if (finding.environmentId) {
          requireEnvironmentScope(session, finding.environmentId);
        }
        return finding;
      }
      case DaemonMethods.explainFinding: {
        requirePermission(session, "findings:read");
        const findingId = requireString(p, "findingId");
        const finding = await this.findingService.get(findingId);
        if (!finding) {
          throw new DaemonAppError("NOT_FOUND", "Finding not found");
        }
        requireProjectScope(session, finding.projectId);
        if (finding.environmentId) {
          requireEnvironmentScope(session, finding.environmentId);
        }
        return {
          finding,
          explanation: finding.summary,
          evidence: finding.evidence,
        };
      }
      case DaemonMethods.getFindingSummary: {
        requirePermission(session, "findings:read");
        const projectId = requireString(p, "projectId");
        requireProjectScope(session, projectId);
        if (
          session.client?.permissionProfileId !== "administrator" &&
          session.environmentScopes &&
          session.environmentScopes !== "*"
        ) {
          const findings = filterEnvironmentScoped(
            await this.findingService.list({ projectId }),
            session,
          );
          return {
            projectId,
            total: findings.length,
            byStatus: countBy(findings, "status"),
            bySeverity: countBy(findings, "severity"),
          };
        }
        return this.findingSummary.getProjectSummary(projectId);
      }
      case DaemonMethods.acknowledgeFinding:
      case DaemonMethods.dismissFinding:
      case DaemonMethods.suppressFinding:
      case DaemonMethods.reopenFinding:
        return this.mutateFinding(session, method, p, correlationId);
      case DaemonMethods.scanFindings: {
        requirePermission(session, "findings:scan");
        const projectId = requireString(p, "projectId");
        requireProjectScope(session, projectId);
        const idempotencyKey = optionalString(p, "idempotencyKey");
        if (idempotencyKey) {
          const existing = this.control.getOperationByIdempotency(idempotencyKey);
          if (existing) {
            if (
              existing.projectId !== projectId ||
              existing.type !== "findings_scan" ||
              actorIdentity(existing.actor) !== actorIdentity(session.actor)
            ) {
              throw new DaemonAppError(
                "VALIDATION_FAILED",
                "Idempotency key was already used for a different request",
              );
            }
            return existing;
          }
        }
        const op = this.control.createOperation({
          projectId,
          type: "findings_scan",
          actor: session.actor,
          correlationId,
          idempotencyKey,
        });
        this.control.updateOperation(op.id, {
          status: "running",
          startedAt: new Date().toISOString(),
        });
        this.events.emit({
          type: "operation_started",
          projectId,
          actor: session.actor,
          correlationId,
          payload: { operationId: op.id, type: op.type },
        });
        // Deterministic local scan: re-list findings (evaluators run via seed/fixture).
        const findings = await this.findingService.list({ projectId });
        const completed = this.control.updateOperation(op.id, {
          status: "succeeded",
          finishedAt: new Date().toISOString(),
          progress: { percent: 100, message: "Scan complete" },
          resultSummary: { findings: findings.length },
        });
        this.events.emit({
          type: "operation_completed",
          projectId,
          actor: session.actor,
          correlationId,
          payload: { operationId: op.id, status: "succeeded" },
        });
        return completed;
      }

      case DaemonMethods.generatePlanFromFinding:
        return this.generatePlanFromFinding(session, p, correlationId);
      case DaemonMethods.generateChangePlan:
        return this.generateChangePlan(session, p, correlationId);
      case DaemonMethods.listChangePlans: {
        requirePermission(session, "plans:read");
        const projectId = requireString(p, "projectId");
        requireProjectScope(session, projectId);
        const plans = await this.changePlanService.listByProjectId(projectId);
        return filterEnvironmentScoped(plans, session);
      }
      case DaemonMethods.getChangePlan: {
        requirePermission(session, "plans:read");
        const planId = requireString(p, "changePlanId");
        const plan = await this.changePlanService.getById(planId);
        if (!plan) return null;
        requireProjectScope(session, plan.projectId);
        if (plan.environmentId) {
          requireEnvironmentScope(session, plan.environmentId);
        }
        return plan;
      }
      case DaemonMethods.approveChangePlan:
        return this.approveChangePlan(session, p, correlationId);
      case DaemonMethods.applyChangePlan:
        return this.applyChangePlan(session, p, correlationId);
      case DaemonMethods.verifyChangePlan:
        return this.verifyChangePlan(session, p, correlationId);
      case DaemonMethods.retryFailedChange:
        return this.retryFailedChange(session, p, correlationId);
      case DaemonMethods.rejectChangePlan: {
        requirePermission(session, "plans:approve");
        const planId = requireString(p, "changePlanId");
        const plan = await this.changePlanService.requireCurrent(planId);
        requireProjectScope(session, plan.projectId);
        if (plan.environmentId) {
          requireEnvironmentScope(session, plan.environmentId);
        }
        await this.changeApprovalService.reject({
          changePlanId: planId,
          rejectedBy: toPluginActor(session.actor),
          reason: optionalString(p, "reason") ?? "Rejected via daemon",
        });
        return { changePlanId: planId, status: "rejected" };
      }

      case DaemonMethods.listOperations: {
        requirePermission(session, "operations:read");
        const projectId = optionalString(p, "projectId");
        if (projectId) requireProjectScope(session, projectId);
        const operations = this.control.listOperations({
          projectId,
          status: optionalString(p, "status") as never,
        });
        return filterProjectAndEnvironmentScoped(operations, session);
      }
      case DaemonMethods.getOperation: {
        requirePermission(session, "operations:read");
        const operationId = requireString(p, "operationId");
        const op = this.control.getOperation(operationId);
        if (op?.projectId) requireProjectScope(session, op.projectId);
        return op;
      }
      case DaemonMethods.cancelOperation: {
        requirePermission(session, "operations:cancel");
        const operationId = requireString(p, "operationId");
        const op = this.control.getOperation(operationId);
        if (!op) throw new DaemonAppError("NOT_FOUND", "Operation not found");
        if (op.projectId) requireProjectScope(session, op.projectId);
        if (["succeeded", "failed", "cancelled"].includes(op.status)) {
          return op;
        }
        if (op.type === "change_apply" && op.status === "running") {
          throw new DaemonAppError(
            "CANCELLED",
            "Cannot cancel apply after provider mutation may have started; verify instead",
            { retryable: false },
          );
        }
        return this.control.updateOperation(operationId, {
          status: "cancelled",
          finishedAt: new Date().toISOString(),
        });
      }

      case DaemonMethods.listApprovals: {
        requirePermission(session, "plans:approve");
        const approvals = this.control.listApprovals({
          status: optionalString(p, "status") as never,
          projectId: optionalString(p, "projectId"),
        });
        return filterProjectAndEnvironmentScoped(approvals, session);
      }
      case DaemonMethods.decideApproval:
        return this.decideApproval(session, p, correlationId);

      case DaemonMethods.createMcpClient:
        return this.createMcpClient(session, p);
      case DaemonMethods.listMcpClients: {
        requirePermission(session, "mcp_clients:manage");
        return this.control.listClients().map((c) => ({
          ...c,
          connected: [...this.connectedSessions.values()].some(
            (s) => s.client?.id === c.id,
          ),
        }));
      }
      case DaemonMethods.getMcpClient: {
        requirePermission(session, "mcp_clients:manage");
        return this.control.getClient(requireString(p, "clientId"));
      }
      case DaemonMethods.revokeMcpClient: {
        requirePermission(session, "mcp_clients:manage");
        const clientId = requireString(p, "clientId");
        const client = this.control.getClient(clientId);
        if (!client) throw new DaemonAppError("NOT_FOUND", "Client not found");
        client.status = "revoked";
        this.control.saveClient(client);
        this.credentials.revoke(clientId);
        this.events.emit({
          type: "mcp_client_changed",
          actor: session.actor,
          payload: { clientId, action: "revoked" },
        });
        return { clientId, status: "revoked" };
      }
      case DaemonMethods.rotateMcpClientCredential: {
        requirePermission(session, "mcp_clients:manage");
        const clientId = requireString(p, "clientId");
        const credential = this.credentials.rotate(clientId);
        return { clientId, credential };
      }
      case DaemonMethods.updateMcpClient: {
        requirePermission(session, "mcp_clients:manage");
        const clientId = requireString(p, "clientId");
        const client = this.control.getClient(clientId);
        if (!client) throw new DaemonAppError("NOT_FOUND", "Client not found");
        if (typeof p.name === "string") client.name = p.name;
        if (typeof p.permissionProfileId === "string") {
          client.permissionProfileId = p.permissionProfileId;
        }
        if (Array.isArray(p.projectScopes)) {
          client.projectScopes = p.projectScopes as string[];
        }
        if (Array.isArray(p.environmentScopes)) {
          client.environmentScopes = p.environmentScopes as string[];
        }
        if (p.approvalPolicy && typeof p.approvalPolicy === "object") {
          client.approvalPolicy = p.approvalPolicy as McpApprovalPolicy;
        }
        this.control.saveClient(client);
        return client;
      }
      case DaemonMethods.getMcpClientScope:
        return {
          clientId: session.client?.id,
          permissions: [...session.permissions],
          projectScopes: session.projectScopes,
          environmentScopes: session.environmentScopes,
          profileId: session.client?.permissionProfileId,
        };
      case DaemonMethods.listAvailableCapabilities:
        return {
          permissions: [...session.permissions],
          methods: Object.values(DaemonMethods),
        };
      case DaemonMethods.listMcpAuditEvents: {
        requirePermission(session, "mcp_clients:manage");
        return this.control.listAuditEvents(
          typeof p.limit === "number" ? p.limit : 100,
        );
      }

      case DaemonMethods.listIntegrations: {
        requirePermission(session, "integrations:read");
        const projectId = optionalString(p, "projectId");
        if (projectId) requireProjectScope(session, projectId);
        await this.exampleLocalHost.ensureReconciled();
        const connections = projectId
          ? await this.exampleLocalHost.connections.listByProjectId(projectId)
          : (
              await Promise.all(
                (
                  await this.projectService.list({ includeArchived: true })
                ).map((project) =>
                  this.exampleLocalHost.connections.listByProjectId(project.id),
                ),
              )
            ).flat();
        return connections.map((connection) => ({
          id: connection.id,
          pluginId: connection.pluginId,
          projectId: connection.projectId,
          name: connection.name,
          status: connection.status,
          lastSuccessfulSyncAt: connection.lastSuccessfulSyncAt,
        }));
      }
      case DaemonMethods.getIntegration: {
        requirePermission(session, "integrations:read");
        const integrationId = requireString(p, "integrationId");
        const connection =
          await this.exampleLocalHost.connections.getById(integrationId);
        if (!connection) return null;
        if (connection.projectId) {
          requireProjectScope(session, connection.projectId);
        }
        return connection;
      }
      case DaemonMethods.listIntegrationResources: {
        requirePermission(session, "integrations:read");
        const integrationId = requireString(p, "integrationId");
        const connection =
          await this.exampleLocalHost.connections.getById(integrationId);
        if (!connection) return [];
        if (connection.projectId) {
          requireProjectScope(session, connection.projectId);
        }
        return this.exampleLocalHost.discovery.listByConnectionId(integrationId);
      }
      case DaemonMethods.getIntegrationHealth: {
        requirePermission(session, "integrations:read");
        const integrationId = requireString(p, "integrationId");
        const connection =
          await this.exampleLocalHost.connections.getById(integrationId);
        if (!connection) {
          throw new DaemonAppError("NOT_FOUND", "Integration not found");
        }
        if (connection.projectId) {
          requireProjectScope(session, connection.projectId);
        }
        return {
          integrationId,
          status: connection.status,
          lastSuccessfulSyncAt: connection.lastSuccessfulSyncAt,
          lastErrorCode: connection.lastErrorCode,
        };
      }
      case DaemonMethods.getEnvironmentResources: {
        requirePermission(session, "integrations:read");
        const projectId = requireString(p, "projectId");
        const environmentId = requireString(p, "environmentId");
        requireProjectScope(session, projectId);
        requireEnvironmentScope(session, environmentId);
        await this.requireEnvironmentInProject(projectId, environmentId);
        const bindings =
          await this.exampleLocalHost.bindings.listByProjectId(projectId);
        return bindings.filter(
          (binding) =>
            !binding.environmentId || binding.environmentId === environmentId,
        );
      }
      case DaemonMethods.syncProject:
      case DaemonMethods.syncEnvironment:
      case DaemonMethods.syncIntegration: {
        requirePermission(session, "integrations:sync");
        const projectId = requireString(p, "projectId");
        requireProjectScope(session, projectId);
        const environmentId = optionalString(p, "environmentId");
        if (environmentId) {
          requireEnvironmentScope(session, environmentId);
          await this.requireEnvironmentInProject(projectId, environmentId);
        }
        const integrationId = optionalString(p, "integrationId");
        return this.syncExampleLocal(
          session,
          projectId,
          environmentId,
          integrationId,
          correlationId,
        );
      }
      case DaemonMethods.inspectResource: {
        requirePermission(session, "resources:read");
        const resourceId = requireString(p, "resourceId");
        const { binding } =
          await this.exampleLocalHost.requireBinding(resourceId);
        requireProjectScope(session, binding.projectId);
        if (binding.environmentId) {
          requireEnvironmentScope(session, binding.environmentId);
        }
        const observed = await this.exampleLocalHost.inspectBinding(
          resourceId,
          toPluginExecutionActor(session.actor),
        );
        return { resourceBindingId: resourceId, observed };
      }
      case DaemonMethods.listPlugins: {
        requirePermission(session, "plugins:read");
        return this.exampleLocalHost.listPluginStatus();
      }
      case DaemonMethods.listPluginActions: {
        requirePermission(session, "plugins:read");
        // Extension point: structured plugin MCP actions (none registered yet).
        return [];
      }

      default:
        throw new DaemonAppError("METHOD_NOT_FOUND", `Unknown method: ${method}`);
    }
  }

  private async requireEnvironmentInProject(
    projectId: string,
    environmentId: string,
  ): Promise<void> {
    const environment = await this.environmentService.getById(environmentId);
    if (!environment || environment.projectId !== projectId) {
      throw new DaemonAppError(
        "ENVIRONMENT_SCOPE_DENIED",
        "Environment does not belong to the requested project",
      );
    }
  }

  private async requireConfigurationKeyInProject(
    projectId: string,
    configurationKeyId: string,
  ): Promise<void> {
    const key = await this.configurationService.getKey(configurationKeyId);
    if (!key || key.projectId !== projectId) {
      throw new DaemonAppError(
        "PROJECT_SCOPE_DENIED",
        "Configuration key does not belong to the requested project",
      );
    }
  }

  private async setConfigurationValue(
    session: SessionContext,
    p: Record<string, unknown>,
    correlationId: string,
    sensitive: boolean,
  ) {
    const configurationKeyId = requireString(p, "configurationKeyId");
    const environmentId = requireString(p, "environmentId");
    const projectId = requireString(p, "projectId");
    requireProjectScope(session, projectId);
    requireEnvironmentScope(session, environmentId);
    await this.requireEnvironmentInProject(projectId, environmentId);
    await this.requireConfigurationKeyInProject(projectId, configurationKeyId);
    const expectedRevision =
      typeof p.expectedRevision === "number" ? p.expectedRevision : undefined;

    let desiredValue: string | undefined;
    let secretValueRef: string | undefined;
    let valueFingerprint: string | undefined;

    if (sensitive) {
      const secret = requireString(p, "secretValue");
      const stored = this.secrets.put(secret);
      secretValueRef = stored.ref;
      valueFingerprint = stored.fingerprint;
    } else {
      desiredValue = requireString(p, "value");
    }

    let saved: Awaited<ReturnType<ConfigurationDesiredStateService["saveDesired"]>>;
    try {
      saved = await this.desiredStateService.saveDesired({
        configurationKeyId,
        environmentId,
        projectId,
        desiredValue,
        secretValueRef,
        valueFingerprint,
        updatedBy: actorToConfigActor(session.actor),
        expectedRevision,
      });
    } catch (error) {
      if (secretValueRef) {
        this.secrets.delete(secretValueRef);
      }
      throw error;
    }

    this.events.emit({
      type: "configuration_changed",
      projectId,
      actor: session.actor,
      correlationId,
      payload: {
        configurationKeyId,
        environmentId,
        action: "saved",
        revision: saved.revision,
        sensitive,
      },
    });

    // Re-evaluate related findings: if desired matches a DEBUG_MODE false pattern, resolve.
    await this.reevaluateFindingsAfterConfigSave(
      projectId,
      environmentId,
      configurationKeyId,
      desiredValue,
      session.actor,
      correlationId,
    );

    return {
      saved: sanitizeDesired(saved),
      remoteStateAffected: false,
      findingsReevaluated: true,
    };
  }

  private async reevaluateFindingsAfterConfigSave(
    projectId: string,
    environmentId: string,
    configurationKeyId: string,
    desiredValue: string | undefined,
    actor: RayvanActor,
    correlationId: string,
  ): Promise<void> {
    const findings = await this.findingService.list({
      projectId,
      environmentId,
      statuses: ["open"],
    });
    for (const finding of findings) {
      const matchesEvidence = finding.evidence?.some(
        (e) =>
          e.type === "configuration_comparison" &&
          e.configurationKeyId === configurationKeyId,
      );
      const matchesKey =
        finding.title.includes("DEBUG_MODE") ||
        matchesEvidence ||
        finding.fingerprint.includes(configurationKeyId);
      if (!matchesKey) continue;
      // Desired saved but not applied → keep open with updated evidence; mark pending apply.
      this.events.emit({
        type: "finding_changed",
        projectId,
        actor,
        correlationId,
        payload: {
          findingId: finding.id,
          environmentId,
          action: "desired_updated",
          desiredValuePresent: desiredValue !== undefined,
        },
      });
    }
  }

  private async mutateFinding(
    session: SessionContext,
    method: string,
    p: Record<string, unknown>,
    correlationId: string,
  ) {
    requirePermission(session, "findings:manage");
    const findingId = requireString(p, "findingId");
    const finding = await this.findingService.get(findingId);
    if (!finding) throw new DaemonAppError("NOT_FOUND", "Finding not found");
    requireProjectScope(session, finding.projectId);
    if (finding.environmentId) {
      requireEnvironmentScope(session, finding.environmentId);
    }
    const actor = toFindingActor(session.actor);
    let updated;
    if (method === DaemonMethods.acknowledgeFinding) {
      updated = await this.findingService.acknowledge(
        findingId,
        actor,
        optionalString(p, "comment"),
      );
    } else if (method === DaemonMethods.dismissFinding) {
      updated = await this.findingService.dismiss(
        findingId,
        actor,
        optionalString(p, "reason"),
      );
    } else if (method === DaemonMethods.suppressFinding) {
      updated = await this.findingService.suppress(findingId, actor, {
        preset: "7d",
        reason: optionalString(p, "reason"),
      });
    } else if (method === DaemonMethods.reopenFinding) {
      updated = await this.findingService.reopen(findingId, actor);
    } else {
      throw new DaemonAppError(
        "VALIDATION_FAILED",
        `Unsupported finding mutation: ${method}`,
      );
    }
    this.events.emit({
      type: "finding_changed",
      projectId: finding.projectId,
      actor: session.actor,
      correlationId,
      payload: {
        findingId,
        environmentId: finding.environmentId,
        status: updated.status,
      },
    });
    return updated;
  }

  private async generatePlanFromFinding(
    session: SessionContext,
    p: Record<string, unknown>,
    correlationId: string,
  ) {
    requirePermission(session, "plans:create");
    const findingId = requireString(p, "findingId");
    const finding = await this.findingService.get(findingId);
    if (!finding) throw new DaemonAppError("NOT_FOUND", "Finding not found");
    requireProjectScope(session, finding.projectId);
    if (finding.environmentId) {
      requireEnvironmentScope(session, finding.environmentId);
    }

    const host = await this.exampleLocalHost.ensureProjectConnection(
      finding.projectId,
    );
    const binding =
      host.bindings.find((item) => item.id === finding.resourceBindingId) ??
      host.bindings[0];
    if (!binding) {
      throw new DaemonAppError(
        "NOT_FOUND",
        "No example-local resource binding is available to plan against",
      );
    }

    const observed = await this.exampleLocalHost.inspectBinding(
      binding.id,
      toPluginExecutionActor(session.actor),
    );
    const observedPort =
      typeof observed.attributes.port === "number"
        ? observed.attributes.port
        : 3000;
    const desiredPort = observedPort === 3000 ? 3010 : 3000;

    const generated = await this.generateChangePlan(
      session,
      {
        projectId: finding.projectId,
        environmentId: finding.environmentId,
        resourceBindingId: binding.id,
        desiredAttributes: { port: desiredPort },
        summaryHint: `Resolve finding: ${finding.title}`,
      },
      correlationId,
    );
    const planId =
      generated &&
      typeof generated === "object" &&
      "plan" in generated &&
      generated.plan &&
      typeof generated.plan === "object" &&
      "id" in generated.plan &&
      typeof generated.plan.id === "string"
        ? generated.plan.id
        : undefined;
    if (planId) {
      await this.findingService.attachChangePlan(findingId, planId);
    }
    return generated;
  }

  private async generateChangePlan(
    session: SessionContext,
    p: Record<string, unknown>,
    correlationId: string,
  ) {
    requirePermission(session, "plans:create");
    const projectId = requireString(p, "projectId");
    requireProjectScope(session, projectId);
    const environmentId = optionalString(p, "environmentId");
    if (environmentId) {
      requireEnvironmentScope(session, environmentId);
      await this.requireEnvironmentInProject(projectId, environmentId);
    }

    const resourceBindingId = optionalString(p, "resourceBindingId");
    const desiredAttributes =
      p.desiredAttributes &&
      typeof p.desiredAttributes === "object" &&
      !Array.isArray(p.desiredAttributes)
        ? (p.desiredAttributes as Record<string, unknown>)
        : undefined;

    const host = await this.exampleLocalHost.ensureProjectConnection(projectId);
    const binding = resourceBindingId
      ? host.bindings.find((item) => item.id === resourceBindingId)
      : host.bindings[0];
    if (!binding) {
      throw new DaemonAppError(
        "NOT_FOUND",
        resourceBindingId
          ? `Resource binding not found: ${resourceBindingId}`
          : "No resource binding available; sync the example-local connection first",
      );
    }
    if (binding.projectId !== projectId) {
      throw new DaemonAppError(
        "PROJECT_SCOPE_DENIED",
        "Resource binding does not belong to the requested project",
      );
    }

    const op = this.control.createOperation({
      projectId,
      type: "change_plan_generation",
      actor: session.actor,
      correlationId,
    });
    this.control.updateOperation(op.id, {
      status: "running",
      startedAt: new Date().toISOString(),
    });

    try {
      const actor = toPluginExecutionActor(session.actor);
      const observed = await this.exampleLocalHost.inspectBinding(
        binding.id,
        actor,
      );
      const observedPort =
        typeof observed.attributes.port === "number"
          ? observed.attributes.port
          : 3000;
      const attributes =
        desiredAttributes ??
        ({
          port: observedPort === 3000 ? 3010 : 3000,
        } satisfies Record<string, unknown>);

      let planPayload;
      try {
        const planned = await this.exampleLocalHost.planForBinding({
          resourceBindingId: binding.id,
          desiredAttributes: attributes,
          actor,
        });
        planPayload = {
          ...planned.plan,
          id: `plan_${randomUUID()}`,
          resourceId: binding.id,
          summary: optionalString(p, "summaryHint") ?? planned.plan.summary,
        };
      } catch (error) {
        if (!(error instanceof DaemonAppError)) throw error;
        const before =
          typeof observed.attributes.port === "number"
            ? observed.attributes.port
            : undefined;
        const after =
          typeof attributes.port === "number" ? attributes.port : undefined;
        planPayload = {
          id: `plan_${randomUUID()}`,
          pluginId: binding.pluginId,
          resourceId: binding.id,
          summary:
            optionalString(p, "summaryHint") ??
            `Local plan for ${binding.displayName ?? binding.id}`,
          operations:
            before !== undefined && after !== undefined && before !== after
              ? [
                  {
                    id: "set-port",
                    type: "update_attribute",
                    description: `Change port from ${before} to ${after}`,
                    path: "attributes.port",
                    before,
                    after,
                    requiresApproval: true,
                    destructive: false,
                  },
                ]
              : [],
          warnings: [
            `Plugin plan unavailable (${error.message}); used structured local plan`,
          ],
          destructive: false,
        };
      }

      try {
        const existingDesired =
          await this.exampleLocalHost.resourceState.getDesired(binding.id);
        await this.exampleLocalHost.resourceState.saveDesired({
          projectId,
          environmentId: environmentId ?? binding.environmentId,
          resourceBindingId: binding.id,
          pluginId: binding.pluginId,
          connectionId: binding.connectionId,
          state: { attributes },
          schemaVersion: "1.0.0",
          createdBy: actor,
          expectedRevision: existingDesired?.revision,
        });
      } catch {
        // Desired-state persistence is best-effort for the in-memory plugin domain.
      }

      const plan = await this.changePlanService.create({
        pluginId: binding.pluginId,
        connectionId: binding.connectionId,
        projectId,
        environmentId: environmentId ?? binding.environmentId,
        resourceBindingId: binding.id,
        plan: planPayload,
        createdBy: toPluginActor(session.actor),
      });

      const completed = this.control.updateOperation(op.id, {
        status: "succeeded",
        finishedAt: new Date().toISOString(),
        resultSummary: { changePlanId: plan.id },
      });
      this.db.raw
        .prepare(`UPDATE operations SET change_plan_id = ? WHERE id = ?`)
        .run(plan.id, op.id);
      this.events.emit({
        type: "operation_completed",
        projectId,
        actor: session.actor,
        correlationId,
        payload: { operationId: op.id, changePlanId: plan.id },
      });
      return { operation: completed, plan };
    } catch (error) {
      this.control.updateOperation(op.id, {
        status: "failed",
        finishedAt: new Date().toISOString(),
        safeError: {
          code: "PROVIDER_OPERATION_FAILED",
          message:
            error instanceof Error ? error.message : "Plan generation failed",
          retryable: true,
        },
      });
      throw error;
    }
  }

  private async approveChangePlan(
    session: SessionContext,
    p: Record<string, unknown>,
    correlationId: string,
  ) {
    requirePermission(session, "plans:approve");
    const planId = requireString(p, "changePlanId");
    const plan = await this.changePlanService.requireCurrent(planId);
    requireProjectScope(session, plan.projectId);
    if (plan.environmentId) {
      requireEnvironmentScope(session, plan.environmentId);
    }

    const policy = session.client?.approvalPolicy ?? {
      type: "client_may_approve" as const,
      allowDestructive: false,
    };

    if (policy.type === "always_require_desktop_approval") {
      const approval = this.control.createApproval({
        projectId: plan.projectId,
        changePlanId: plan.id,
        requestedBy: session.actor,
        type: plan.plan.destructive ? "destructive_operation" : "remote_apply",
        summary: `Approve change plan ${plan.id}: ${plan.plan.summary}`,
        safeDetails: {
          changePlanId: plan.id,
          environmentId: plan.environmentId ?? null,
        },
      });
      this.events.emit({
        type: "approval_requested",
        projectId: plan.projectId,
        actor: session.actor,
        correlationId,
        payload: { approvalId: approval.id, changePlanId: plan.id },
      });
      return {
        status: "pending_approval",
        approvalId: approval.id,
        changePlanId: plan.id,
      };
    }

    if (
      policy.type === "allow_preapproved_scope" &&
      !policy.projectIds.includes(plan.projectId)
    ) {
      throw new DaemonAppError(
        "APPROVAL_DENIED",
        "Plan project is outside preapproved scope",
      );
    }
    if (
      policy.type === "allow_preapproved_scope" &&
      plan.environmentId &&
      policy.environmentIds &&
      !policy.environmentIds.includes(plan.environmentId)
    ) {
      throw new DaemonAppError(
        "APPROVAL_DENIED",
        "Plan environment is outside preapproved scope",
      );
    }
    if (
      policy.type === "allow_preapproved_scope" &&
      policy.pluginIds &&
      !policy.pluginIds.includes(plan.pluginId)
    ) {
      throw new DaemonAppError(
        "APPROVAL_DENIED",
        "Plan plugin is outside preapproved scope",
      );
    }
    if (
      policy.type === "allow_preapproved_scope" &&
      (!policy.permissions.includes("plans:approve") ||
        !policy.permissions.includes("changes:apply"))
    ) {
      throw new DaemonAppError(
        "APPROVAL_DENIED",
        "Preapproved scope does not authorize plan approval and apply",
      );
    }

    if (plan.plan.destructive) {
      const allowDestructive =
        policy.type === "client_may_approve"
          ? policy.allowDestructive
          : policy.type === "allow_preapproved_scope"
            ? policy.allowDestructive
            : false;
      if (!allowDestructive) {
        throw new DaemonAppError(
          "APPROVAL_DENIED",
          "Destructive plans are not allowed by approval policy",
        );
      }
    }

    await this.changeApprovalService.approve({
      changePlanId: planId,
      approvedOperationIds: plan.plan.operations.map((operation) => operation.id),
      destructiveApproval: plan.plan.destructive,
      approvedBy: toPluginActor(session.actor),
    });
    return { status: "approved", changePlanId: planId };
  }

  private async applyChangePlan(
    session: SessionContext,
    p: Record<string, unknown>,
    correlationId: string,
  ) {
    requirePermission(session, "changes:apply");
    const planId = requireString(p, "changePlanId");
    const plan = await this.changePlanService.requireCurrent(planId);
    requireProjectScope(session, plan.projectId);
    if (plan.environmentId) {
      requireEnvironmentScope(session, plan.environmentId);
    }

    if (plan.pluginId !== "example-local") {
      throw new DaemonAppError(
        "PLUGIN_UNAVAILABLE",
        `No daemon-owned plugin host is available for plugin ${plan.pluginId}`,
        { retryable: true, correlationId },
      );
    }

    const binding = await this.pluginRepos.resourceBindings.getById(
      plan.resourceBindingId,
    );
    if (!binding) {
      throw new DaemonAppError(
        "NOT_FOUND",
        `Resource binding not found for plan: ${plan.resourceBindingId}`,
      );
    }

    const approved =
      await this.changeApprovalService.buildApprovedChangePlan(planId);
    const op = this.control.createOperation({
      projectId: plan.projectId,
      type: "change_apply",
      actor: session.actor,
      correlationId,
      idempotencyKey: optionalString(p, "idempotencyKey"),
    });
    this.control.updateOperation(op.id, {
      status: "running",
      startedAt: new Date().toISOString(),
    });
    this.db.raw
      .prepare(`UPDATE operations SET change_plan_id = ? WHERE id = ?`)
      .run(plan.id, op.id);

    const startedAt = new Date().toISOString();
    await this.changeApprovalService.beginApply(planId);
    const actor = toPluginExecutionActor(session.actor);

    try {
      const execution = await this.exampleLocalHost.applyPlan(
        plan,
        approved,
        actor,
      );
      if (execution.status !== "succeeded") {
        const apply = await this.changeApprovalService.completeApply({
          changePlanId: planId,
          executionId: execution.executionId,
          status: "failed",
          error: execution.error,
          startedAt,
        });
        this.control.updateOperation(op.id, {
          status: "failed",
          finishedAt: new Date().toISOString(),
          safeError: {
            code: "PROVIDER_OPERATION_FAILED",
            message: execution.error?.message ?? "Apply failed",
            retryable: execution.error?.retryable ?? true,
          },
        });
        return { operationId: op.id, apply, status: "failed" as const };
      }

      const apply = await this.changeApprovalService.completeApply({
        changePlanId: planId,
        executionId: execution.executionId,
        status: "succeeded",
        result: execution.data,
        startedAt,
      });
      this.control.updateOperation(op.id, {
        status: "succeeded",
        finishedAt: new Date().toISOString(),
        resultSummary: {
          changePlanId: planId,
          changeApplyId: apply.id,
          appliedOperationIds: execution.data.appliedOperationIds,
        },
      });
      this.events.emit({
        type: "operation_completed",
        projectId: plan.projectId,
        actor: session.actor,
        correlationId,
        payload: {
          operationId: op.id,
          changePlanId: planId,
          changeApplyId: apply.id,
        },
      });
      return {
        operationId: op.id,
        apply,
        result: execution.data,
        status: "succeeded" as const,
      };
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Apply failed unexpectedly";
      await this.changeApprovalService
        .completeApply({
          changePlanId: planId,
          executionId: `failed_${randomUUID()}`,
          status: "failed",
          error: {
            code: "execution_failed",
            message,
            pluginId: plan.pluginId,
            capability: "apply",
            retryable: true,
          },
          startedAt,
        })
        .catch(() => undefined);
      this.control.updateOperation(op.id, {
        status: "failed",
        finishedAt: new Date().toISOString(),
        safeError: {
          code: "PROVIDER_OPERATION_FAILED",
          message,
          retryable: true,
        },
      });
      throw error;
    }
  }

  private async verifyChangePlan(
    session: SessionContext,
    p: Record<string, unknown>,
    correlationId: string,
  ) {
    requirePermission(session, "changes:verify");
    const planId = requireString(p, "changePlanId");
    const plan = await this.changePlanService.getById(planId);
    if (!plan) throw new DaemonAppError("NOT_FOUND", "Change plan not found");
    requireProjectScope(session, plan.projectId);
    if (plan.environmentId) {
      requireEnvironmentScope(session, plan.environmentId);
    }

    if (plan.pluginId !== "example-local") {
      throw new DaemonAppError(
        "PLUGIN_UNAVAILABLE",
        `No daemon-owned plugin host is available for plugin ${plan.pluginId}`,
        { retryable: true, correlationId },
      );
    }

    const applies = await this.pluginRepos.changeApplies.listByPlanId(planId);
    const latestApply = applies.at(-1);
    if (!latestApply?.result) {
      throw new DaemonAppError(
        "VALIDATION_FAILED",
        "Cannot verify: no completed apply result exists for this plan",
      );
    }

    const approved =
      await this.changeApprovalService.buildApprovedChangePlan(planId);
    const execution = await this.exampleLocalHost.verifyPlan(
      plan,
      approved,
      latestApply.result,
      toPluginExecutionActor(session.actor),
    );

    if (execution.status !== "succeeded") {
      const verification = await this.changeApprovalService.recordVerification({
        changeApplyId: latestApply.id,
        executionId: execution.executionId,
        status: "failed",
        error: execution.error,
      });
      return { verification, status: "failed" as const };
    }

    const verification = await this.changeApprovalService.recordVerification({
      changeApplyId: latestApply.id,
      executionId: execution.executionId,
      status: execution.data.ok ? "verified" : "not_verified",
      result: execution.data,
    });

    const resolvedFindingIds: string[] = [];
    if (execution.data.ok) {
      const linked = await this.findingService.list({
        projectId: plan.projectId,
        statuses: ["open", "acknowledged"],
      });
      for (const finding of linked) {
        if (finding.changePlanId !== planId) continue;
        const resolved = await this.findingService.resolve(
          finding.id,
          toFindingActor(session.actor),
          `Verified change plan ${planId}`,
        );
        resolvedFindingIds.push(resolved.id);
        this.events.emit({
          type: "finding_changed",
          projectId: plan.projectId,
          actor: session.actor,
          correlationId,
          payload: {
            findingId: resolved.id,
            environmentId: resolved.environmentId,
            status: resolved.status,
            action: "resolved_by_verification",
            changePlanId: planId,
          },
        });
      }
    }

    this.events.emit({
      type: "operation_completed",
      projectId: plan.projectId,
      actor: session.actor,
      correlationId,
      payload: {
        changePlanId: planId,
        changeApplyId: latestApply.id,
        verificationId: verification.id,
        verified: execution.data.ok,
        resolvedFindingIds,
      },
    });
    return {
      verification,
      result: execution.data,
      status: execution.data.ok ? ("verified" as const) : ("not_verified" as const),
      resolvedFindingIds,
    };
  }

  private async retryFailedChange(
    session: SessionContext,
    p: Record<string, unknown>,
    correlationId: string,
  ) {
    requirePermission(session, "changes:apply");
    const planId = requireString(p, "changePlanId");
    const plan = await this.changePlanService.getById(planId);
    if (!plan) throw new DaemonAppError("NOT_FOUND", "Change plan not found");
    requireProjectScope(session, plan.projectId);
    if (plan.environmentId) {
      requireEnvironmentScope(session, plan.environmentId);
    }

    if (plan.status === "applying") {
      throw new DaemonAppError(
        "VALIDATION_FAILED",
        "Cannot retry an interrupted apply; verify the provider state first",
        { retryable: false },
      );
    }
    if (plan.status !== "failed") {
      throw new DaemonAppError(
        "VALIDATION_FAILED",
        `Only failed change plans can be retried (status=${plan.status})`,
      );
    }

    const applies = await this.pluginRepos.changeApplies.listByPlanId(planId);
    const latestApply = applies.at(-1);
    if (!latestApply) {
      throw new DaemonAppError(
        "VALIDATION_FAILED",
        "Cannot retry: no apply record exists for this failed plan",
      );
    }

    const interruptMessage =
      latestApply.error?.message?.includes("interrupted") ||
      latestApply.error?.message?.includes("verification required before retry");
    if (interruptMessage) {
      throw new DaemonAppError(
        "VALIDATION_FAILED",
        "Interrupted applies cannot be blindly retried; verify first",
        { retryable: false },
      );
    }
    if (latestApply.status !== "failed") {
      throw new DaemonAppError(
        "VALIDATION_FAILED",
        "retryFailed only retries applies that completed with status failed",
      );
    }

    await this.pluginRepos.changePlans.setStatus(planId, "approved");
    return this.applyChangePlan(
      session,
      { changePlanId: planId, idempotencyKey: optionalString(p, "idempotencyKey") },
      correlationId,
    );
  }

  private async syncExampleLocal(
    session: SessionContext,
    projectId: string,
    environmentId: string | undefined,
    integrationId: string | undefined,
    correlationId: string,
  ) {
    const actor = toPluginExecutionActor(session.actor);
    let host;
    if (integrationId) {
      const connection =
        await this.exampleLocalHost.connections.getById(integrationId);
      if (!connection || connection.projectId !== projectId) {
        throw new DaemonAppError(
          "NOT_FOUND",
          "Integration connection not found for project",
        );
      }
      if (connection.pluginId !== "example-local") {
        throw new DaemonAppError(
          "PLUGIN_UNAVAILABLE",
          `No daemon-owned provider runtime for plugin ${connection.pluginId}`,
          { retryable: true, correlationId },
        );
      }
      host = await this.exampleLocalHost.syncConnection(connection, actor);
    } else {
      host = await this.exampleLocalHost.ensureProjectConnection(projectId);
    }

    if (environmentId) {
      // Environment sync uses the same discovery; bindings may be environment-scoped later.
      void environmentId;
    }

    this.events.emit({
      type: "operation_completed",
      projectId,
      actor: session.actor,
      correlationId,
      payload: {
        type: "sync",
        connectionId: host.connection.id,
        discovered: host.discovered.length,
        bindings: host.bindings.length,
      },
    });

    return {
      connectionId: host.connection.id,
      pluginId: host.connection.pluginId,
      discovered: host.discovered,
      bindings: host.bindings,
      remoteStateAffected: false,
    };
  }

  private async setConfigurationTargets(
    session: SessionContext,
    p: Record<string, unknown>,
    correlationId: string,
  ) {
    requirePermission(session, "configuration:write");
    const projectId = requireString(p, "projectId");
    requireProjectScope(session, projectId);
    const configurationKeyId = requireString(p, "configurationKeyId");
    await this.requireConfigurationKeyInProject(projectId, configurationKeyId);
    const resourceBindingId = requireString(p, "resourceBindingId");
    await this.exampleLocalHost.requireBinding(resourceBindingId);

    const occurrenceIds = Array.isArray(p.occurrenceIds)
      ? (p.occurrenceIds as unknown[]).filter(
          (id): id is string => typeof id === "string" && id.length > 0,
        )
      : undefined;

    const occurrences = occurrenceIds
      ? await Promise.all(
          occurrenceIds.map(async (id) => {
            const occurrence = await this.configurationService.getOccurrence(id);
            if (!occurrence || occurrence.projectId !== projectId) {
              throw new DaemonAppError(
                "NOT_FOUND",
                `Configuration occurrence not found: ${id}`,
              );
            }
            if (occurrence.configurationKeyId !== configurationKeyId) {
              throw new DaemonAppError(
                "VALIDATION_FAILED",
                "Occurrence does not belong to the requested configuration key",
              );
            }
            return occurrence;
          }),
        )
      : (
          await this.configurationService.listOccurrencesByKey(configurationKeyId)
        ).filter((occurrence) => occurrence.projectId === projectId);

    if (occurrences.length === 0) {
      throw new DaemonAppError(
        "NOT_FOUND",
        "No configuration occurrences available to target",
      );
    }

    const updated = [];
    for (const occurrence of occurrences) {
      updated.push(
        await this.configurationService.updateOccurrence(occurrence.id, {
          resourceBindingId,
        }),
      );
    }

    this.events.emit({
      type: "configuration_changed",
      projectId,
      actor: session.actor,
      correlationId,
      payload: {
        configurationKeyId,
        action: "targets_set",
        resourceBindingId,
        occurrenceIds: updated.map((item) => item.id),
      },
    });
    return {
      configurationKeyId,
      resourceBindingId,
      occurrences: updated.map((occurrence) =>
        sanitizeOccurrence(occurrence, false),
      ),
    };
  }

  private async removeConfigurationTarget(
    session: SessionContext,
    p: Record<string, unknown>,
    correlationId: string,
  ) {
    requirePermission(session, "configuration:write");
    const projectId = requireString(p, "projectId");
    requireProjectScope(session, projectId);
    const occurrenceId = requireString(p, "occurrenceId");
    const occurrence =
      await this.configurationService.getOccurrence(occurrenceId);
    if (!occurrence || occurrence.projectId !== projectId) {
      throw new DaemonAppError("NOT_FOUND", "Configuration occurrence not found");
    }
    const updated = await this.configurationService.updateOccurrence(
      occurrenceId,
      { resourceBindingId: null },
    );
    this.events.emit({
      type: "configuration_changed",
      projectId,
      actor: session.actor,
      correlationId,
      payload: {
        configurationKeyId: occurrence.configurationKeyId,
        action: "target_removed",
        occurrenceId,
      },
    });
    return { occurrence: sanitizeOccurrence(updated, false) };
  }

  private async adoptDiscoveredConfiguration(
    session: SessionContext,
    p: Record<string, unknown>,
    correlationId: string,
  ) {
    requirePermission(session, "configuration:write");
    const projectId = requireString(p, "projectId");
    requireProjectScope(session, projectId);
    const occurrenceId = requireString(p, "occurrenceId");
    const occurrence =
      await this.configurationService.getOccurrence(occurrenceId);
    if (!occurrence || occurrence.projectId !== projectId) {
      throw new DaemonAppError("NOT_FOUND", "Configuration occurrence not found");
    }
    const key = await this.configurationService.getKey(
      occurrence.configurationKeyId,
    );
    if (!key) {
      throw new DaemonAppError("NOT_FOUND", "Configuration key not found");
    }

    const managed = await this.configurationService.upsertKeyByName(
      projectId,
      key.name,
      {
        description: key.description,
        valueType: key.valueType,
        required: key.required,
        sensitive: key.sensitive,
        source: "manual",
      },
    );

    const clearedIgnore = await this.configurationService.updateOccurrence(
      occurrenceId,
      {
        scope: occurrence.scope === "ignored" ? null : occurrence.scope,
        resourceBindingId:
          optionalString(p, "resourceBindingId") ?? occurrence.resourceBindingId,
      },
    );

    const environmentId =
      optionalString(p, "environmentId") ?? occurrence.environmentId;
    let desired = null;
    if (
      environmentId &&
      occurrence.observedValue !== undefined &&
      !key.sensitive
    ) {
      requireEnvironmentScope(session, environmentId);
      await this.requireEnvironmentInProject(projectId, environmentId);
      const existingDesired = await this.desiredStateService.getDesired(
        managed.id,
        environmentId,
      );
      desired = await this.desiredStateService.saveDesired({
        configurationKeyId: managed.id,
        environmentId,
        projectId,
        desiredValue: occurrence.observedValue,
        updatedBy: actorToConfigActor(session.actor),
        expectedRevision: existingDesired?.revision,
      });
    }

    this.events.emit({
      type: "configuration_changed",
      projectId,
      actor: session.actor,
      correlationId,
      payload: {
        configurationKeyId: managed.id,
        occurrenceId,
        action: "adopted",
      },
    });

    return {
      key: sanitizeKey(managed),
      occurrence: sanitizeOccurrence(clearedIgnore, managed.sensitive),
      desired: desired ? sanitizeDesired(desired) : null,
    };
  }

  private async ignoreDiscoveredConfiguration(
    session: SessionContext,
    p: Record<string, unknown>,
    correlationId: string,
  ) {
    requirePermission(session, "configuration:write");
    const projectId = requireString(p, "projectId");
    requireProjectScope(session, projectId);
    const occurrenceId = requireString(p, "occurrenceId");
    const occurrence =
      await this.configurationService.getOccurrence(occurrenceId);
    if (!occurrence || occurrence.projectId !== projectId) {
      throw new DaemonAppError("NOT_FOUND", "Configuration occurrence not found");
    }
    const updated = await this.configurationService.updateOccurrence(
      occurrenceId,
      { scope: "ignored" },
    );
    this.events.emit({
      type: "configuration_changed",
      projectId,
      actor: session.actor,
      correlationId,
      payload: {
        configurationKeyId: occurrence.configurationKeyId,
        occurrenceId,
        action: "ignored",
      },
    });
    return { occurrence: sanitizeOccurrence(updated, false) };
  }

  private async decideApproval(
    session: SessionContext,
    p: Record<string, unknown>,
    correlationId: string,
  ) {
    requirePermission(session, "plans:approve");
    const approvalId = requireString(p, "approvalId");
    const decision = requireString(p, "decision");
    const approval = this.control.getApproval(approvalId);
    if (!approval) throw new DaemonAppError("NOT_FOUND", "Approval not found");
    requireProjectScope(session, approval.projectId);
    if (
      approval.requestedBy.type === "mcp_client" &&
      session.client?.type !== "desktop"
    ) {
      throw new DaemonAppError(
        "APPROVAL_DENIED",
        "This approval request must be decided by Rayvan Desktop",
      );
    }
    if (approval.status !== "pending") return approval;

    approval.status = decision === "approved" ? "approved" : "denied";
    approval.decidedAt = new Date().toISOString();
    approval.decidedBy = session.actor;
    this.control.updateApproval(approval);

    if (approval.status === "approved" && approval.changePlanId) {
      const plan = await this.changePlanService.requireCurrent(approval.changePlanId);
      await this.changeApprovalService.approve({
        changePlanId: approval.changePlanId,
        approvedOperationIds: plan.plan.operations.map((operation) => operation.id),
        destructiveApproval: plan.plan.destructive,
        approvedBy: toPluginActor(session.actor),
      });
    }

    this.events.emit({
      type: "approval_decided",
      projectId: approval.projectId,
      actor: session.actor,
      correlationId,
      payload: {
        approvalId,
        status: approval.status,
        changePlanId: approval.changePlanId,
      },
    });
    return approval;
  }

  private createMcpClient(session: SessionContext, p: Record<string, unknown>) {
    requirePermission(session, "mcp_clients:manage");
    const name = requireString(p, "name");
    const permissionProfileId = requireString(
      p,
      "permissionProfileId",
    ) as BuiltInPermissionProfileId;
    if (permissionProfileId === "administrator") {
      // Allowed but not default — caller must choose explicitly.
    }
    const projectScopes = Array.isArray(p.projectScopes)
      ? (p.projectScopes as string[])
      : [];
    if (projectScopes.length === 0) {
      throw new DaemonAppError("VALIDATION_FAILED", "projectScopes must be non-empty");
    }
    const environmentScopes = Array.isArray(p.environmentScopes)
      ? (p.environmentScopes as string[])
      : undefined;
    const approvalPolicy =
      (p.approvalPolicy as McpApprovalPolicy | undefined) ??
      ({
        type: "client_may_approve",
        allowDestructive: false,
      } satisfies McpApprovalPolicy);

    const id = `mcp_${randomUUID().replace(/-/g, "").slice(0, 16)}`;
    const credential = this.credentials.issue(id);
    const client: LocalClientRecord = {
      id,
      name,
      type: "mcp",
      status: "active",
      permissionProfileId,
      projectScopes,
      environmentScopes,
      approvalPolicy,
      createdAt: new Date().toISOString(),
      credentialReferenceId: `cred_ref_${id}`,
    };
    this.control.saveClient(client);
    this.events.emit({
      type: "mcp_client_changed",
      actor: session.actor,
      payload: { clientId: id, action: "created" },
    });

    const mcpConfig = {
      mcpServers: {
        rayvan: {
          command: "rayvan-mcp",
          args: ["serve", "--client-id", id],
        },
      },
    };

    return { client, credential, mcpConfig };
  }

  private provisionSystemClient(
    id: string,
    name: string,
    type: "desktop" | "cli",
  ): void {
    const existing = this.control.getClient(id);
    if (!existing) {
      this.control.saveClient({
        id,
        name,
        type,
        status: "active",
        permissionProfileId: "administrator",
        projectScopes: [],
        approvalPolicy: {
          type: "client_may_approve",
          allowDestructive: false,
        },
        createdAt: new Date().toISOString(),
        credentialReferenceId: `keyring:${id}`,
      });
    }
    // Re-issue when keyring has an orphan secret or the hash metadata is gone.
    if (!this.credentials.resolve(id)) {
      this.credentials.issue(id);
    }
  }

  close(): void {
    this.db.close();
  }
}

// --- helpers ---

function asRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  return value as Record<string, unknown>;
}

function requireString(p: Record<string, unknown>, key: string): string {
  const value = p[key];
  if (typeof value !== "string" || value.length === 0) {
    throw new DaemonAppError(
      "VALIDATION_FAILED",
      `Missing or invalid string param: ${key}`,
    );
  }
  return value;
}

function optionalString(p: Record<string, unknown>, key: string): string | undefined {
  const value = p[key];
  return typeof value === "string" ? value : undefined;
}

function requireNumber(p: Record<string, unknown>, key: string): number {
  const value = p[key];
  if (typeof value !== "number") {
    throw new DaemonAppError(
      "VALIDATION_FAILED",
      `Missing or invalid number param: ${key}`,
    );
  }
  return value;
}

function readString(params: unknown, key: string): string | undefined {
  const p = asRecord(params);
  return optionalString(p, key);
}

function actorIdentity(actor: RayvanActor): string {
  return actor.type === "plugin"
    ? `plugin:${actor.pluginId}`
    : `${actor.type}:${actor.id}`;
}

function countBy<T extends Record<K, string>, K extends keyof T>(
  values: T[],
  key: K,
): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const value of values) {
    counts[value[key]] = (counts[value[key]] ?? 0) + 1;
  }
  return counts;
}

function resolveProjectScopes(
  client: { permissionProfileId: string; projectScopes: string[] } | undefined,
): string[] | "*" {
  if (!client) return "*";
  // Built-in desktop/cli clients are provisioned with an empty scope list and
  // an administrator profile — treat that as unrestricted project access.
  if (
    client.permissionProfileId === "administrator" ||
    client.projectScopes.length === 0
  ) {
    return "*";
  }
  return client.projectScopes;
}

function filterProjects<T extends { id: string }>(
  projects: T[],
  session: SessionContext,
): T[] {
  if (
    session.projectScopes === "*" ||
    session.client?.permissionProfileId === "administrator"
  ) {
    return projects;
  }
  return projects.filter((p) => session.projectScopes.includes(p.id));
}

function filterEnvironments<T extends { id: string }>(
  environments: T[],
  session: SessionContext,
): T[] {
  if (
    session.client?.permissionProfileId === "administrator" ||
    !session.environmentScopes ||
    session.environmentScopes === "*"
  ) {
    return environments;
  }
  return environments.filter(
    (environment) =>
      session.environmentScopes !== "*" &&
      session.environmentScopes?.includes(environment.id),
  );
}

function filterEnvironmentScoped<T extends { environmentId?: string }>(
  values: T[],
  session: SessionContext,
): T[] {
  if (
    session.client?.permissionProfileId === "administrator" ||
    !session.environmentScopes ||
    session.environmentScopes === "*"
  ) {
    return values;
  }
  return values.filter(
    (value) =>
      !value.environmentId ||
      (session.environmentScopes !== "*" &&
        session.environmentScopes?.includes(value.environmentId)),
  );
}

function filterProjectAndEnvironmentScoped<
  T extends { projectId?: string; environmentId?: string },
>(values: T[], session: SessionContext): T[] {
  if (session.client?.permissionProfileId === "administrator") {
    return values;
  }
  return filterEnvironmentScoped(
    values.filter(
      (value) =>
        !value.projectId ||
        session.projectScopes === "*" ||
        session.projectScopes.includes(value.projectId),
    ),
    session,
  );
}

function sanitizeKey<T>(key: T): T {
  return key;
}

function sanitizeOccurrence<
  T extends {
    observedValue?: string;
    secretValueRef?: string;
  },
>(occurrence: T, sensitive: boolean): Omit<T, "secretValueRef"> {
  const { secretValueRef: _secretValueRef, ...safe } = occurrence;
  void _secretValueRef;
  return {
    ...safe,
    ...(sensitive ? { observedValue: undefined } : {}),
  };
}

function sanitizeDesired<
  T extends { desiredValue?: string | null; secretValueRef?: string | null },
>(
  desired: T,
): Omit<T, "desiredValue" | "secretValueRef"> & {
  desiredValue?: string | null;
  hasSecret: boolean;
} {
  const hasSecret = Boolean(desired.secretValueRef);
  const { secretValueRef: _secretValueRef, desiredValue, ...safe } = desired;
  void _secretValueRef;
  return {
    ...safe,
    desiredValue: hasSecret ? null : desiredValue,
    hasSecret,
  };
}

function actorToConfigActor(actor: RayvanActor) {
  if (actor.type === "mcp_client") {
    return { kind: "mcp_agent" as const, id: actor.id };
  }
  if (actor.type === "user" || actor.type === "desktop") {
    return { kind: "user" as const, id: actor.id };
  }
  return {
    kind: "system" as const,
    id: actor.type === "plugin" ? actor.pluginId : actor.id,
  };
}

function toPluginActor(actor: RayvanActor) {
  if (actor.type === "mcp_client") {
    return { type: "mcp_agent" as const, id: actor.id };
  }
  if (actor.type === "user" || actor.type === "desktop") {
    return { type: "user" as const, id: actor.id };
  }
  return { type: "system" as const, id: "daemon" };
}

function toFindingActor(actor: RayvanActor) {
  if (actor.type === "mcp_client") {
    return { kind: "mcp_agent" as const, id: actor.id };
  }
  if (actor.type === "user" || actor.type === "desktop") {
    return { kind: "user" as const, id: actor.id };
  }
  return {
    kind: "system" as const,
    id: actor.type === "plugin" ? actor.pluginId : actor.id,
  };
}
