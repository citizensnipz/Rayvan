import Database from "better-sqlite3";

import { MIGRATION_VERSION, PROJECTS_TABLE_MIGRATION } from "./migrations.js";
import { ProjectPersistenceError } from "../projects/errors.js";

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
      const currentVersion = this.getCurrentVersion();
      if (currentVersion < MIGRATION_VERSION) {
        this.db.exec(PROJECTS_TABLE_MIGRATION);
        this.db
          .prepare(
            "INSERT OR REPLACE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
          )
          .run(MIGRATION_VERSION, new Date().toISOString());
      }
    } catch (error) {
      throw new ProjectPersistenceError("Failed to initialize local database", error);
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
