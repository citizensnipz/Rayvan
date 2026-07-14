import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import Database from "better-sqlite3";
import { describe, expect, it } from "vitest";

import { LocalDatabaseConnection } from "../src/database/connection.js";
import { MIGRATION_VERSION } from "../src/database/migrations.js";
import {
  SqliteInstalledPluginRepository,
  SqlitePluginConnectionRepository,
} from "../src/plugins/sqlite/repositories.js";

function canUseBetterSqlite3(): boolean {
  try {
    const database = new Database(":memory:");
    database.close();
    return true;
  } catch {
    return false;
  }
}

const describeSqlite = canUseBetterSqlite3() ? describe : describe.skip;

describeSqlite("plugin sqlite migration smoke", () => {
  it("applies v3 and persists installed plugins", async () => {
    const directory = mkdtempSync(join(tmpdir(), "rayvan-plugin-db-"));
    const dbPath = join(directory, "rayvan.db");

    try {
      const connection = new LocalDatabaseConnection(dbPath);
      const version = connection.raw
        .prepare("SELECT MAX(version) AS version FROM schema_migrations")
        .get() as { version: number };
      expect(version.version).toBe(MIGRATION_VERSION);

      const tables = connection.raw
        .prepare(
          "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'plugin_%'",
        )
        .all() as Array<{ name: string }>;
      expect(tables.map((row) => row.name)).toEqual(
        expect.arrayContaining([
          "plugin_installed",
          "plugin_connections",
          "plugin_execution_history",
        ]),
      );

      const installed = new SqliteInstalledPluginRepository(connection);
      await installed.save({
        id: "inst-1",
        pluginId: "example.local",
        pluginVersion: "0.0.1",
        manifestVersion: "0.0.1",
        rayvanApiVersion: "1",
        source: { type: "built_in" },
        status: "installed",
        enabled: true,
        installedAt: "2026-01-01T00:00:00.000Z",
        updatedAt: "2026-01-01T00:00:00.000Z",
        manifestSnapshot: {
          id: "example.local",
          name: "Example",
          version: "0.0.1",
          publisher: "rayvan",
          rayvanApiVersion: "1",
          capabilities: ["discover"],
          permissions: [],
          resourceTypes: [],
        },
      });

      const connections = new SqlitePluginConnectionRepository(connection);
      await connections.save({
        id: "conn-1",
        installedPluginId: "inst-1",
        pluginId: "example.local",
        name: "Default",
        status: "connected",
        metadata: {},
        schemaVersion: "1",
        createdAt: "2026-01-01T00:00:00.000Z",
        updatedAt: "2026-01-01T00:00:00.000Z",
      });

      connection.close();

      const reopened = new LocalDatabaseConnection(dbPath);
      const loaded = await new SqliteInstalledPluginRepository(
        reopened,
      ).getByPluginId("example.local");
      const loadedConn = await new SqlitePluginConnectionRepository(
        reopened,
      ).getById("conn-1");
      reopened.close();

      expect(loaded?.pluginId).toBe("example.local");
      expect(loadedConn?.name).toBe("Default");
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });
});
