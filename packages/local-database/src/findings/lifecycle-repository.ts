import type { FindingLifecycleEventRecord } from "@rayvan/core";
import type { FindingLifecycleEventRepository } from "@rayvan/findings-engine";

import type { LocalDatabaseConnection } from "../database/connection.js";
import { FindingPersistenceError } from "./errors.js";
import {
  lifecycleEventToBindParams,
  mapLifecycleEventRow,
  type FindingLifecycleEventRow,
} from "./mappers.js";

const SELECT_COLUMNS = `id, finding_id, project_id, type, actor_json, created_at, previous_status, next_status, reason, metadata_json`;

const INSERT_SQL = `INSERT INTO finding_lifecycle_events (
  id, finding_id, project_id, type, actor_json, created_at,
  previous_status, next_status, reason, metadata_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`;

export class InMemoryFindingLifecycleEventRepository
  implements FindingLifecycleEventRepository
{
  readonly events: FindingLifecycleEventRecord[] = [];

  async append(event: FindingLifecycleEventRecord): Promise<void> {
    this.events.push(structuredClone(event));
  }

  async listByFindingId(
    findingIdValue: string,
  ): Promise<FindingLifecycleEventRecord[]> {
    return this.events
      .filter((event) => String(event.findingId) === findingIdValue)
      .map((event) => structuredClone(event))
      .sort((left, right) => left.createdAt.localeCompare(right.createdAt));
  }
}

export class SqliteFindingLifecycleEventRepository
  implements FindingLifecycleEventRepository
{
  constructor(private readonly connection: LocalDatabaseConnection) {}

  async append(event: FindingLifecycleEventRecord): Promise<void> {
    try {
      this.connection.raw
        .prepare(INSERT_SQL)
        .run(...lifecycleEventToBindParams(event));
    } catch (error) {
      throw new FindingPersistenceError(
        "Failed to append finding lifecycle event",
        error,
      );
    }
  }

  async listByFindingId(
    findingIdValue: string,
  ): Promise<FindingLifecycleEventRecord[]> {
    try {
      const rows = this.connection.raw
        .prepare(
          `SELECT ${SELECT_COLUMNS} FROM finding_lifecycle_events
           WHERE finding_id = ?
           ORDER BY created_at ASC`,
        )
        .all(findingIdValue) as FindingLifecycleEventRow[];
      return rows.map(mapLifecycleEventRow);
    } catch (error) {
      throw new FindingPersistenceError(
        "Failed to list finding lifecycle events",
        error,
      );
    }
  }
}
