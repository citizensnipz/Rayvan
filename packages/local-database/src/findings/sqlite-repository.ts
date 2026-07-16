import type { FindingRecord } from "@rayvan/core";
import type {
  FindingQuery,
  FindingRepository,
} from "@rayvan/findings-engine";

import type { LocalDatabaseConnection } from "../database/connection.js";
import { FindingPersistenceError } from "./errors.js";
import {
  findingRecordToBindParams,
  mapFindingRow,
  type FindingRow,
} from "./mappers.js";
import { filterFindingRecords } from "./memory-repository.js";

const SELECT_COLUMNS = `id, project_id, rule_id, source_json, category, severity, title, summary,
  description, status, fingerprint, fingerprint_version, environment_id, integration_id,
  connection_id, discovered_resource_id, resource_binding_id, configuration_key_id,
  change_plan_id, deployment_id, evidence_json, remediation_json, first_detected_at,
  last_detected_at, occurrence_count, acknowledged_at, acknowledged_by_json, dismissed_at,
  dismissed_by_json, dismissal_reason, resolved_at, resolution_json, suppressed_until,
  last_evaluation_run_id, metadata_json, schema_version`;

const UPSERT_SQL = `INSERT INTO findings (
  id, project_id, rule_id, source_json, category, severity, title, summary,
  description, status, fingerprint, fingerprint_version, environment_id, integration_id,
  connection_id, discovered_resource_id, resource_binding_id, configuration_key_id,
  change_plan_id, deployment_id, evidence_json, remediation_json, first_detected_at,
  last_detected_at, occurrence_count, acknowledged_at, acknowledged_by_json, dismissed_at,
  dismissed_by_json, dismissal_reason, resolved_at, resolution_json, suppressed_until,
  last_evaluation_run_id, metadata_json, schema_version
) VALUES (
  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
)
ON CONFLICT(id) DO UPDATE SET
  project_id = excluded.project_id,
  rule_id = excluded.rule_id,
  source_json = excluded.source_json,
  category = excluded.category,
  severity = excluded.severity,
  title = excluded.title,
  summary = excluded.summary,
  description = excluded.description,
  status = excluded.status,
  fingerprint = excluded.fingerprint,
  fingerprint_version = excluded.fingerprint_version,
  environment_id = excluded.environment_id,
  integration_id = excluded.integration_id,
  connection_id = excluded.connection_id,
  discovered_resource_id = excluded.discovered_resource_id,
  resource_binding_id = excluded.resource_binding_id,
  configuration_key_id = excluded.configuration_key_id,
  change_plan_id = excluded.change_plan_id,
  deployment_id = excluded.deployment_id,
  evidence_json = excluded.evidence_json,
  remediation_json = excluded.remediation_json,
  first_detected_at = excluded.first_detected_at,
  last_detected_at = excluded.last_detected_at,
  occurrence_count = excluded.occurrence_count,
  acknowledged_at = excluded.acknowledged_at,
  acknowledged_by_json = excluded.acknowledged_by_json,
  dismissed_at = excluded.dismissed_at,
  dismissed_by_json = excluded.dismissed_by_json,
  dismissal_reason = excluded.dismissal_reason,
  resolved_at = excluded.resolved_at,
  resolution_json = excluded.resolution_json,
  suppressed_until = excluded.suppressed_until,
  last_evaluation_run_id = excluded.last_evaluation_run_id,
  metadata_json = excluded.metadata_json,
  schema_version = excluded.schema_version`;

export class SqliteFindingRepository implements FindingRepository {
  constructor(private readonly connection: LocalDatabaseConnection) {}

  async getById(id: string): Promise<FindingRecord | undefined> {
    try {
      const row = this.connection.raw
        .prepare(`SELECT ${SELECT_COLUMNS} FROM findings WHERE id = ?`)
        .get(id) as FindingRow | undefined;
      return row ? mapFindingRow(row) : undefined;
    } catch (error) {
      throw new FindingPersistenceError("Failed to get finding", error);
    }
  }

  async getByFingerprint(
    projectIdValue: string,
    fingerprint: string,
  ): Promise<FindingRecord | undefined> {
    try {
      const row = this.connection.raw
        .prepare(
          `SELECT ${SELECT_COLUMNS} FROM findings
           WHERE project_id = ? AND fingerprint = ?`,
        )
        .get(projectIdValue, fingerprint) as FindingRow | undefined;
      return row ? mapFindingRow(row) : undefined;
    } catch (error) {
      throw new FindingPersistenceError(
        "Failed to get finding by fingerprint",
        error,
      );
    }
  }

  async list(query: FindingQuery): Promise<FindingRecord[]> {
    try {
      const rows = this.connection.raw
        .prepare(
          `SELECT ${SELECT_COLUMNS} FROM findings WHERE project_id = ?`,
        )
        .all(String(query.projectId)) as FindingRow[];
      return filterFindingRecords(rows.map(mapFindingRow), query);
    } catch (error) {
      throw new FindingPersistenceError("Failed to list findings", error);
    }
  }

  async save(record: FindingRecord): Promise<void> {
    try {
      this.connection.raw
        .prepare(UPSERT_SQL)
        .run(...findingRecordToBindParams(record));
    } catch (error) {
      throw new FindingPersistenceError("Failed to save finding", error);
    }
  }

  async saveMany(records: FindingRecord[]): Promise<void> {
    try {
      const upsert = this.connection.raw.prepare(UPSERT_SQL);
      const runAll = this.connection.raw.transaction(
        (items: FindingRecord[]) => {
          for (const record of items) {
            upsert.run(...findingRecordToBindParams(record));
          }
        },
      );
      runAll(records);
    } catch (error) {
      throw new FindingPersistenceError("Failed to save findings", error);
    }
  }

  /** Persist a finding and append a lifecycle event in one transaction. */
  saveWithLifecycleEvent(
    record: FindingRecord,
    eventParams: unknown[],
  ): void {
    try {
      const upsert = this.connection.raw.prepare(UPSERT_SQL);
      const append = this.connection.raw.prepare(
        `INSERT INTO finding_lifecycle_events (
          id, finding_id, project_id, type, actor_json, created_at,
          previous_status, next_status, reason, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      );
      const run = this.connection.raw.transaction(() => {
        upsert.run(...findingRecordToBindParams(record));
        append.run(...eventParams);
      });
      run();
    } catch (error) {
      throw new FindingPersistenceError(
        "Failed to save finding with lifecycle event",
        error,
      );
    }
  }
}
