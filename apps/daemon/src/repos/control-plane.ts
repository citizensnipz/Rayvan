import { randomUUID } from "node:crypto";

import type {
  ApprovalRequestRecord,
  LocalClientRecord,
  McpApprovalPolicy,
  OperationProgress,
  OperationRecord,
  OperationStatus,
  OperationType,
  RayvanActor,
} from "@rayvan/daemon-contracts";
import type { LocalDatabaseConnection } from "@rayvan/local-database/sqlite";

function asJson<T>(value: string): T {
  return JSON.parse(value) as T;
}

export class ControlPlaneRepository {
  constructor(private readonly db: LocalDatabaseConnection) {}

  // --- Local clients ---

  listClients(): LocalClientRecord[] {
    const rows = this.db.raw
      .prepare(`SELECT * FROM local_clients ORDER BY created_at DESC`)
      .all() as Array<Record<string, unknown>>;
    return rows.map((row) => this.mapClient(row));
  }

  getClient(id: string): LocalClientRecord | null {
    const row = this.db.raw
      .prepare(`SELECT * FROM local_clients WHERE id = ?`)
      .get(id) as Record<string, unknown> | undefined;
    return row ? this.mapClient(row) : null;
  }

  saveClient(client: LocalClientRecord): void {
    this.db.raw
      .prepare(
        `INSERT INTO local_clients (
          id, name, type, status, permission_profile_id, project_scopes_json,
          environment_scopes_json, approval_policy_json, credential_reference_id,
          created_at, last_connected_at, last_activity_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          name = excluded.name,
          type = excluded.type,
          status = excluded.status,
          permission_profile_id = excluded.permission_profile_id,
          project_scopes_json = excluded.project_scopes_json,
          environment_scopes_json = excluded.environment_scopes_json,
          approval_policy_json = excluded.approval_policy_json,
          credential_reference_id = excluded.credential_reference_id,
          last_connected_at = excluded.last_connected_at,
          last_activity_at = excluded.last_activity_at`,
      )
      .run(
        client.id,
        client.name,
        client.type,
        client.status,
        client.permissionProfileId,
        JSON.stringify(client.projectScopes),
        client.environmentScopes ? JSON.stringify(client.environmentScopes) : null,
        JSON.stringify(client.approvalPolicy),
        client.credentialReferenceId,
        client.createdAt,
        client.lastConnectedAt ?? null,
        client.lastActivityAt ?? null,
      );
  }

  touchClient(id: string, connected: boolean): void {
    const now = new Date().toISOString();
    if (connected) {
      this.db.raw
        .prepare(
          `UPDATE local_clients SET last_connected_at = ?, last_activity_at = ? WHERE id = ?`,
        )
        .run(now, now, id);
    } else {
      this.db.raw
        .prepare(`UPDATE local_clients SET last_activity_at = ? WHERE id = ?`)
        .run(now, id);
    }
  }

  // --- Operations ---

  createOperation(input: {
    projectId?: string;
    type: OperationType;
    actor: RayvanActor;
    correlationId: string;
    idempotencyKey?: string;
    changePlanId?: string;
  }): OperationRecord {
    if (input.idempotencyKey) {
      const existing = this.getOperationByIdempotency(input.idempotencyKey);
      if (existing) {
        return existing;
      }
    }
    const now = new Date().toISOString();
    const record: OperationRecord = {
      id: `op_${randomUUID()}`,
      projectId: input.projectId,
      type: input.type,
      status: "queued",
      actor: input.actor,
      correlationId: input.correlationId,
      idempotencyKey: input.idempotencyKey,
      changePlanId: input.changePlanId,
    };
    this.db.raw
      .prepare(
        `INSERT INTO operations (
          id, project_id, type, status, actor_json, progress_json, started_at,
          finished_at, correlation_id, idempotency_key, safe_error_json,
          result_summary_json, approval_request_id, change_plan_id, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?, NULL, NULL, NULL, ?, ?, ?)`,
      )
      .run(
        record.id,
        record.projectId ?? null,
        record.type,
        record.status,
        JSON.stringify(record.actor),
        record.correlationId,
        record.idempotencyKey ?? null,
        record.changePlanId ?? null,
        now,
        now,
      );
    return record;
  }

  getOperation(id: string): OperationRecord | null {
    const row = this.db.raw.prepare(`SELECT * FROM operations WHERE id = ?`).get(id) as
      Record<string, unknown> | undefined;
    return row ? this.mapOperation(row) : null;
  }

  getOperationByIdempotency(key: string): OperationRecord | null {
    const row = this.db.raw
      .prepare(`SELECT * FROM operations WHERE idempotency_key = ?`)
      .get(key) as Record<string, unknown> | undefined;
    return row ? this.mapOperation(row) : null;
  }

  listOperations(filter?: {
    projectId?: string;
    status?: OperationStatus;
    limit?: number;
  }): OperationRecord[] {
    let sql = `SELECT * FROM operations WHERE 1=1`;
    const params: unknown[] = [];
    if (filter?.projectId) {
      sql += ` AND project_id = ?`;
      params.push(filter.projectId);
    }
    if (filter?.status) {
      sql += ` AND status = ?`;
      params.push(filter.status);
    }
    sql += ` ORDER BY created_at DESC LIMIT ?`;
    params.push(filter?.limit ?? 100);
    const rows = this.db.raw.prepare(sql).all(...params) as Array<
      Record<string, unknown>
    >;
    return rows.map((row) => this.mapOperation(row));
  }

  updateOperation(
    id: string,
    patch: {
      status?: OperationStatus;
      progress?: OperationProgress;
      startedAt?: string;
      finishedAt?: string;
      safeError?: OperationRecord["safeError"];
      resultSummary?: Record<string, unknown>;
      approvalRequestId?: string;
    },
  ): OperationRecord {
    const current = this.getOperation(id);
    if (!current) {
      throw new Error(`Operation not found: ${id}`);
    }
    const next: OperationRecord = {
      ...current,
      status: patch.status ?? current.status,
      progress: patch.progress ?? current.progress,
      startedAt: patch.startedAt ?? current.startedAt,
      finishedAt: patch.finishedAt ?? current.finishedAt,
      safeError: patch.safeError ?? current.safeError,
      resultSummary: patch.resultSummary ?? current.resultSummary,
      approvalRequestId: patch.approvalRequestId ?? current.approvalRequestId,
    };
    this.db.raw
      .prepare(
        `UPDATE operations SET
          status = ?, progress_json = ?, started_at = ?, finished_at = ?,
          safe_error_json = ?, result_summary_json = ?, approval_request_id = ?,
          updated_at = ?
        WHERE id = ?`,
      )
      .run(
        next.status,
        next.progress ? JSON.stringify(next.progress) : null,
        next.startedAt ?? null,
        next.finishedAt ?? null,
        next.safeError ? JSON.stringify(next.safeError) : null,
        next.resultSummary ? JSON.stringify(next.resultSummary) : null,
        next.approvalRequestId ?? null,
        new Date().toISOString(),
        id,
      );
    return next;
  }

  listIncompleteOperations(): OperationRecord[] {
    const rows = this.db.raw
      .prepare(
        `SELECT * FROM operations WHERE status IN ('queued', 'running', 'waiting_for_approval')`,
      )
      .all() as Array<Record<string, unknown>>;
    return rows.map((row) => this.mapOperation(row));
  }

  // --- Approvals ---

  createApproval(
    input: Omit<ApprovalRequestRecord, "id" | "createdAt" | "status"> & {
      status?: ApprovalRequestRecord["status"];
    },
  ): ApprovalRequestRecord {
    const record: ApprovalRequestRecord = {
      id: `apr_${randomUUID()}`,
      createdAt: new Date().toISOString(),
      status: input.status ?? "pending",
      ...input,
    };
    this.db.raw
      .prepare(
        `INSERT INTO approval_requests (
          id, project_id, operation_id, change_plan_id, requested_by_json, type,
          summary, safe_details_json, status, created_at, expires_at, decided_at, decided_by_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        record.id,
        record.projectId,
        record.operationId ?? null,
        record.changePlanId ?? null,
        JSON.stringify(record.requestedBy),
        record.type,
        record.summary,
        JSON.stringify(record.safeDetails),
        record.status,
        record.createdAt,
        record.expiresAt ?? null,
        record.decidedAt ?? null,
        record.decidedBy ? JSON.stringify(record.decidedBy) : null,
      );
    return record;
  }

  getApproval(id: string): ApprovalRequestRecord | null {
    const row = this.db.raw
      .prepare(`SELECT * FROM approval_requests WHERE id = ?`)
      .get(id) as Record<string, unknown> | undefined;
    return row ? this.mapApproval(row) : null;
  }

  listApprovals(filter?: {
    status?: ApprovalRequestRecord["status"];
    projectId?: string;
  }): ApprovalRequestRecord[] {
    let sql = `SELECT * FROM approval_requests WHERE 1=1`;
    const params: unknown[] = [];
    if (filter?.status) {
      sql += ` AND status = ?`;
      params.push(filter.status);
    }
    if (filter?.projectId) {
      sql += ` AND project_id = ?`;
      params.push(filter.projectId);
    }
    sql += ` ORDER BY created_at DESC LIMIT 200`;
    const rows = this.db.raw.prepare(sql).all(...params) as Array<
      Record<string, unknown>
    >;
    return rows.map((row) => this.mapApproval(row));
  }

  updateApproval(record: ApprovalRequestRecord): void {
    this.db.raw
      .prepare(
        `UPDATE approval_requests SET
          status = ?, decided_at = ?, decided_by_json = ?,
          safe_details_json = ?
        WHERE id = ?`,
      )
      .run(
        record.status,
        record.decidedAt ?? null,
        record.decidedBy ? JSON.stringify(record.decidedBy) : null,
        JSON.stringify(record.safeDetails),
        record.id,
      );
  }

  // --- Audit ---

  insertAudit(event: {
    clientId?: string;
    toolName?: string;
    daemonMethod?: string;
    projectId?: string;
    environmentId?: string;
    startedAt: string;
    finishedAt?: string;
    status: string;
    operationId?: string;
    approvalId?: string;
    affectedObjectIds: string[];
    safeSummary: string;
    errorCode?: string;
    contactedRemote: boolean;
    mutatedRemote: boolean;
    correlationId: string;
  }): void {
    this.db.raw
      .prepare(
        `INSERT INTO mcp_audit_events (
          id, client_id, tool_name, daemon_method, project_id, environment_id,
          started_at, finished_at, status, operation_id, approval_id,
          affected_object_ids_json, safe_summary, error_code, contacted_remote,
          mutated_remote, correlation_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        `aud_${randomUUID()}`,
        event.clientId ?? null,
        event.toolName ?? null,
        event.daemonMethod ?? null,
        event.projectId ?? null,
        event.environmentId ?? null,
        event.startedAt,
        event.finishedAt ?? null,
        event.status,
        event.operationId ?? null,
        event.approvalId ?? null,
        JSON.stringify(event.affectedObjectIds),
        event.safeSummary,
        event.errorCode ?? null,
        event.contactedRemote ? 1 : 0,
        event.mutatedRemote ? 1 : 0,
        event.correlationId,
      );
  }

  listAuditEvents(limit = 100): Array<Record<string, unknown>> {
    return this.db.raw
      .prepare(
        `SELECT id, client_id, tool_name, daemon_method, project_id, environment_id,
                started_at, finished_at, status, operation_id, approval_id,
                affected_object_ids_json, safe_summary, error_code,
                contacted_remote, mutated_remote, correlation_id
         FROM mcp_audit_events ORDER BY started_at DESC LIMIT ?`,
      )
      .all(limit) as Array<Record<string, unknown>>;
  }

  // --- Locks ---

  tryAcquireResourceLock(
    resourceKey: string,
    operationId: string,
    actor: RayvanActor,
  ): boolean {
    try {
      this.db.raw
        .prepare(
          `INSERT INTO resource_locks (id, resource_key, operation_id, holder_actor_json, acquired_at, expires_at)
           VALUES (?, ?, ?, ?, ?, NULL)`,
        )
        .run(
          `lock_${randomUUID()}`,
          resourceKey,
          operationId,
          JSON.stringify(actor),
          new Date().toISOString(),
        );
      return true;
    } catch {
      return false;
    }
  }

  releaseResourceLock(resourceKey: string): void {
    this.db.raw
      .prepare(`DELETE FROM resource_locks WHERE resource_key = ?`)
      .run(resourceKey);
  }

  releaseResourceLocksForOperation(operationId: string): void {
    this.db.raw
      .prepare(`DELETE FROM resource_locks WHERE operation_id = ?`)
      .run(operationId);
  }

  seedPermissionProfiles(): void {
    const now = new Date().toISOString();
    const profiles = [
      ["read_only", "Read only"],
      ["planner", "Planner"],
      ["operator", "Operator"],
      ["administrator", "Administrator"],
    ] as const;
    const insert = this.db.raw.prepare(
      `INSERT OR IGNORE INTO mcp_permission_profiles (id, name, built_in, permissions_json, created_at, updated_at)
       VALUES (?, ?, 1, '[]', ?, ?)`,
    );
    for (const [id, name] of profiles) {
      insert.run(id, name, now, now);
    }
  }

  private mapClient(row: Record<string, unknown>): LocalClientRecord {
    return {
      id: String(row.id),
      name: String(row.name),
      type: row.type as LocalClientRecord["type"],
      status: row.status as LocalClientRecord["status"],
      permissionProfileId: String(row.permission_profile_id),
      projectScopes: asJson<string[]>(String(row.project_scopes_json)),
      environmentScopes: row.environment_scopes_json
        ? asJson<string[]>(String(row.environment_scopes_json))
        : undefined,
      approvalPolicy: asJson<McpApprovalPolicy>(String(row.approval_policy_json)),
      createdAt: String(row.created_at),
      lastConnectedAt: row.last_connected_at
        ? String(row.last_connected_at)
        : undefined,
      lastActivityAt: row.last_activity_at ? String(row.last_activity_at) : undefined,
      credentialReferenceId: String(row.credential_reference_id),
    };
  }

  private mapOperation(row: Record<string, unknown>): OperationRecord {
    return {
      id: String(row.id),
      projectId: row.project_id ? String(row.project_id) : undefined,
      type: row.type as OperationType,
      status: row.status as OperationStatus,
      actor: asJson<RayvanActor>(String(row.actor_json)),
      progress: row.progress_json
        ? asJson<OperationProgress>(String(row.progress_json))
        : undefined,
      startedAt: row.started_at ? String(row.started_at) : undefined,
      finishedAt: row.finished_at ? String(row.finished_at) : undefined,
      correlationId: String(row.correlation_id),
      idempotencyKey: row.idempotency_key ? String(row.idempotency_key) : undefined,
      safeError: row.safe_error_json ? asJson(String(row.safe_error_json)) : undefined,
      resultSummary: row.result_summary_json
        ? asJson(String(row.result_summary_json))
        : undefined,
      approvalRequestId: row.approval_request_id
        ? String(row.approval_request_id)
        : undefined,
      changePlanId: row.change_plan_id ? String(row.change_plan_id) : undefined,
    };
  }

  private mapApproval(row: Record<string, unknown>): ApprovalRequestRecord {
    return {
      id: String(row.id),
      projectId: String(row.project_id),
      operationId: row.operation_id ? String(row.operation_id) : undefined,
      changePlanId: row.change_plan_id ? String(row.change_plan_id) : undefined,
      requestedBy: asJson(String(row.requested_by_json)),
      type: row.type as ApprovalRequestRecord["type"],
      summary: String(row.summary),
      safeDetails: asJson(String(row.safe_details_json)),
      status: row.status as ApprovalRequestRecord["status"],
      createdAt: String(row.created_at),
      expiresAt: row.expires_at ? String(row.expires_at) : undefined,
      decidedAt: row.decided_at ? String(row.decided_at) : undefined,
      decidedBy: row.decided_by_json ? asJson(String(row.decided_by_json)) : undefined,
    };
  }
}
