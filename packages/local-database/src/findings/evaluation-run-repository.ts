import type { FindingEvaluationRunRecord } from "@rayvan/core";
import type { FindingEvaluationRunRepository } from "@rayvan/findings-engine";

import type { LocalDatabaseConnection } from "../database/connection.js";
import { FindingPersistenceError } from "./errors.js";
import {
  evaluationRunToBindParams,
  mapEvaluationRunRow,
  type FindingEvaluationRunRow,
} from "./mappers.js";

const SELECT_COLUMNS = `id, project_id, scope_json, trigger, status, started_at, finished_at,
  evaluators_run, evaluators_failed, findings_created, findings_updated,
  findings_reopened, findings_resolved, safe_errors_json`;

const UPSERT_SQL = `INSERT INTO finding_evaluation_runs (
  id, project_id, scope_json, trigger, status, started_at, finished_at,
  evaluators_run, evaluators_failed, findings_created, findings_updated,
  findings_reopened, findings_resolved, safe_errors_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
  project_id = excluded.project_id,
  scope_json = excluded.scope_json,
  trigger = excluded.trigger,
  status = excluded.status,
  started_at = excluded.started_at,
  finished_at = excluded.finished_at,
  evaluators_run = excluded.evaluators_run,
  evaluators_failed = excluded.evaluators_failed,
  findings_created = excluded.findings_created,
  findings_updated = excluded.findings_updated,
  findings_reopened = excluded.findings_reopened,
  findings_resolved = excluded.findings_resolved,
  safe_errors_json = excluded.safe_errors_json`;

export class InMemoryFindingEvaluationRunRepository
  implements FindingEvaluationRunRepository
{
  readonly byId = new Map<string, FindingEvaluationRunRecord>();

  async save(run: FindingEvaluationRunRecord): Promise<void> {
    this.byId.set(String(run.id), structuredClone(run));
  }

  async getById(
    id: string,
  ): Promise<FindingEvaluationRunRecord | undefined> {
    const run = this.byId.get(id);
    return run ? structuredClone(run) : undefined;
  }

  async listByProject(
    projectIdValue: string,
    limit?: number,
  ): Promise<FindingEvaluationRunRecord[]> {
    const runs = [...this.byId.values()]
      .filter((run) => String(run.projectId) === projectIdValue)
      .sort((left, right) => right.startedAt.localeCompare(left.startedAt));
    const sliced =
      limit === undefined ? runs : runs.slice(0, limit);
    return sliced.map((run) => structuredClone(run));
  }
}

export class SqliteFindingEvaluationRunRepository
  implements FindingEvaluationRunRepository
{
  constructor(private readonly connection: LocalDatabaseConnection) {}

  async save(run: FindingEvaluationRunRecord): Promise<void> {
    try {
      this.connection.raw
        .prepare(UPSERT_SQL)
        .run(...evaluationRunToBindParams(run));
    } catch (error) {
      throw new FindingPersistenceError(
        "Failed to save finding evaluation run",
        error,
      );
    }
  }

  async getById(
    id: string,
  ): Promise<FindingEvaluationRunRecord | undefined> {
    try {
      const row = this.connection.raw
        .prepare(
          `SELECT ${SELECT_COLUMNS} FROM finding_evaluation_runs WHERE id = ?`,
        )
        .get(id) as FindingEvaluationRunRow | undefined;
      return row ? mapEvaluationRunRow(row) : undefined;
    } catch (error) {
      throw new FindingPersistenceError(
        "Failed to get finding evaluation run",
        error,
      );
    }
  }

  async listByProject(
    projectIdValue: string,
    limit?: number,
  ): Promise<FindingEvaluationRunRecord[]> {
    try {
      const sql =
        limit === undefined
          ? `SELECT ${SELECT_COLUMNS} FROM finding_evaluation_runs
             WHERE project_id = ?
             ORDER BY started_at DESC`
          : `SELECT ${SELECT_COLUMNS} FROM finding_evaluation_runs
             WHERE project_id = ?
             ORDER BY started_at DESC
             LIMIT ?`;
      const rows =
        limit === undefined
          ? (this.connection.raw.prepare(sql).all(projectIdValue) as FindingEvaluationRunRow[])
          : (this.connection.raw
              .prepare(sql)
              .all(projectIdValue, limit) as FindingEvaluationRunRow[]);
      return rows.map(mapEvaluationRunRow);
    } catch (error) {
      throw new FindingPersistenceError(
        "Failed to list finding evaluation runs",
        error,
      );
    }
  }
}
