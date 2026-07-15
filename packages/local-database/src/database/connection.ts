import Database from "better-sqlite3";

import { ProjectPersistenceError } from "../projects/errors.js";
import {
  APP_SETTINGS_TABLE_MIGRATION,
  MIGRATION_VERSION,
  PROJECTS_TABLE_MIGRATION,
  V3_PLUGIN_PERSISTENCE_SQL,
  V4_ENVIRONMENTS_AND_CONFIGURATION_SQL,
  V5_DESIRED_APPLIED_CONFIGURATION_SQL,
} from "./migrations.js";

interface MigrationStep {
  version: number;
  sql: string;
}

const MIGRATION_STEPS: MigrationStep[] = [
  { version: 1, sql: PROJECTS_TABLE_MIGRATION },
  { version: 2, sql: APP_SETTINGS_TABLE_MIGRATION },
  { version: 3, sql: V3_PLUGIN_PERSISTENCE_SQL },
  { version: 4, sql: V4_ENVIRONMENTS_AND_CONFIGURATION_SQL },
  { version: 5, sql: V5_DESIRED_APPLIED_CONFIGURATION_SQL },
];

export class LocalDatabaseConnection {
  private readonly db: Database.Database;

  constructor(path: string) {
    try {
      this.db = new Database(path);
      this.db.pragma("journal_mode = WAL");
      this.db.pragma("foreign_keys = ON");
      this.initialize();
    } catch (error) {
      throw new ProjectPersistenceError("Failed to open local database", error);
    }
  }

  get raw(): Database.Database {
    return this.db;
  }

  close(): void {
    this.db.close();
  }

  private initialize(): void {
    try {
      let currentVersion = this.getCurrentVersion();
      for (const migration of MIGRATION_STEPS) {
        if (migration.version <= currentVersion) {
          continue;
        }
        this.db.exec(migration.sql);
        this.db
          .prepare(
            "INSERT OR REPLACE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
          )
          .run(migration.version, new Date().toISOString());
        currentVersion = migration.version;
      }

      if (currentVersion < MIGRATION_VERSION) {
        throw new ProjectPersistenceError(
          `Local database schema version ${currentVersion} is below expected ${MIGRATION_VERSION}`,
        );
      }
    } catch (error) {
      if (error instanceof ProjectPersistenceError) {
        throw error;
      }
      throw new ProjectPersistenceError(
        "Failed to initialize local database",
        error,
      );
    }
  }

  private getCurrentVersion(): number {
    const tableExists = this.db
      .prepare(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'",
      )
      .get();

    if (!tableExists) {
      return 0;
    }

    const row = this.db
      .prepare("SELECT MAX(version) AS version FROM schema_migrations")
      .get() as { version: number | null };

    return row.version ?? 0;
  }
}
