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
import {
  SqliteFindingRepository,
  SqliteFindingLifecycleEventRepository,
  createInMemoryPluginPersistence,
} from "@rayvan/local-database";
import { createPluginExecutionStack } from "@rayvan/plugin-sdk";
import { plugin as exampleLocalPlugin } from "@rayvan/plugin-example-local";
import { MIGRATION_VERSION } from "@rayvan/local-database";

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
  private readonly connectedSessions = new Map<string, SessionContext>();
  private readonly allowUnauthenticatedTestClient: boolean;
  private shuttingDown = false;

  constructor(options: DaemonRuntimeOptions = {}) {
    this.dataDir = options.dataDir ?? defaultRayvanDataDir();
    this.runtimeDir = options.runtimeDir ?? defaultRayvanRuntimeDir();
    this.endpoint = options.endpoint ?? daemonEndpointPath(this.runtimeDir);
    this.allowUnauthenticatedTestClient =
      options.allowUnauthenticatedTestClient === true;
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
    const pluginRepos = createInMemoryPluginPersistence();

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
    this.changePlanService = new ChangePlanService(pluginRepos.changePlans);
    this.changeApprovalService = new ChangeApprovalService(
      pluginRepos.changePlans,
      pluginRepos.changePlanApprovals,
      pluginRepos.changeApplies,
      pluginRepos.changeVerifications,
    );

    this.pluginStack = createPluginExecutionStack({
      plugins: [exampleLocalPlugin],
    });
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
      pluginHostStatus: "unavailable",
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
      session.projectScopes = current.projectScopes;
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
        const updated = await this.projectService.update(projectId, {
          name: optionalString(p, "name"),
          description: optionalString(p, "description"),
        });
        this.events.emit({
          type: "project_changed",
          projectId,
          actor: session.actor,
          correlationId,
          payload: { projectId, action: "updated" },
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
          .filter(
            (occurrence) =>
              keysById.get(occurrence.configurationKeyId)?.source === "discovered",
          )
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
      case DaemonMethods.listChangePlans: {
        requirePermission(session, "plans:read");
        const projectId = requireString(p, "projectId");
        requireProjectScope(session, projectId);
        // ChangePlanService has getById; list via sqlite raw for project
        const rows = this.db.raw
          .prepare(
            `SELECT id, plugin_id AS pluginId, project_id AS projectId,
                    environment_id AS environmentId, status, created_at AS createdAt
             FROM plugin_change_plans WHERE project_id = ? ORDER BY created_at DESC LIMIT 100`,
          )
          .all(projectId) as Array<{
          id: string;
          pluginId: string;
          projectId: string;
          environmentId?: string;
          status: string;
          createdAt: string;
        }>;
        return filterEnvironmentScoped(rows, session);
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

      case DaemonMethods.listIntegrations:
      case DaemonMethods.getIntegration:
      case DaemonMethods.listIntegrationResources:
      case DaemonMethods.getIntegrationHealth:
      case DaemonMethods.getEnvironmentResources: {
        requirePermission(session, "integrations:read");
        const projectId = optionalString(p, "projectId");
        if (projectId) requireProjectScope(session, projectId);
        return [];
      }
      case DaemonMethods.syncProject:
      case DaemonMethods.syncEnvironment:
      case DaemonMethods.syncIntegration: {
        requirePermission(session, "integrations:sync");
        const projectId = requireString(p, "projectId");
        requireProjectScope(session, projectId);
        throw new DaemonAppError(
          "PLUGIN_UNAVAILABLE",
          "No daemon-owned provider discovery runtime is available for this integration",
          { retryable: true, correlationId },
        );
      }
      case DaemonMethods.inspectResource: {
        requirePermission(session, "resources:read");
        throw new DaemonAppError(
          "PLUGIN_UNAVAILABLE",
          "No daemon-owned provider inspection runtime is available for this resource",
          { retryable: true, correlationId },
        );
      }
      case DaemonMethods.listPlugins: {
        requirePermission(session, "plugins:read");
        return [
          {
            pluginId: "example-local",
            status: "unavailable",
            reason: "Out-of-process daemon plugin host is not connected",
          },
        ];
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
    } else {
      throw new DaemonAppError(
        "VALIDATION_FAILED",
        "Reopening findings is not supported by the current lifecycle service",
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

    const op = this.control.createOperation({
      projectId: finding.projectId,
      type: "change_plan_generation",
      actor: session.actor,
      correlationId,
    });
    this.control.updateOperation(op.id, {
      status: "running",
      startedAt: new Date().toISOString(),
    });

    const planId = `plan_${randomUUID()}`;
    const plan = await this.changePlanService.create({
      pluginId: "example-local",
      connectionId: "conn_example_local",
      projectId: finding.projectId,
      environmentId: finding.environmentId,
      resourceBindingId: "binding_example_local",
      plan: {
        id: planId,
        pluginId: "example-local",
        resourceId: "resource_example_local",
        summary: `Resolve finding: ${finding.title}`,
        operations: [
          {
            id: "apply-desired-config",
            type: "update_attribute",
            description: "Apply desired configuration to mock remote",
            path: "attributes.debugMode",
            before: true,
            after: false,
            requiresApproval: true,
            destructive: false,
          },
        ],
        warnings: [],
        destructive: false,
      },
      createdBy: toPluginActor(session.actor),
    });

    this.control.updateOperation(op.id, {
      status: "succeeded",
      finishedAt: new Date().toISOString(),
      resultSummary: { changePlanId: plan.id },
      // store plan id
    });
    const completed = this.control.updateOperation(op.id, {
      resultSummary: { changePlanId: plan.id },
    });
    // patch change_plan_id column
    this.db.raw
      .prepare(`UPDATE operations SET change_plan_id = ? WHERE id = ?`)
      .run(plan.id, op.id);

    this.events.emit({
      type: "operation_completed",
      projectId: finding.projectId,
      actor: session.actor,
      correlationId,
      payload: { operationId: op.id, changePlanId: plan.id },
    });

    return { operation: { ...completed, changePlanId: plan.id }, plan };
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
    throw new DaemonAppError(
      "PLUGIN_UNAVAILABLE",
      "Remote apply is disabled until the daemon-owned out-of-process plugin host is available",
      { retryable: true, correlationId },
    );
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
    throw new DaemonAppError(
      "PLUGIN_UNAVAILABLE",
      "Remote verification is disabled until the daemon-owned out-of-process plugin host is available",
      { retryable: true, correlationId },
    );
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

function filterProjects<T extends { id: string }>(
  projects: T[],
  session: SessionContext,
): T[] {
  if (session.projectScopes === "*") return projects;
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
