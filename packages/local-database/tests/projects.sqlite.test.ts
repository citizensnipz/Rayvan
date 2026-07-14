import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import Database from "better-sqlite3";
import { describe, expect, it } from "vitest";

import { LocalDatabaseConnection } from "../src/database/connection.js";
import { SqliteProjectRepository } from "../src/projects/sqlite-repository.js";

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

describeSqlite("SqliteProjectRepository", () => {
  it("persists data across repository reinitialization", async () => {
    const directory = mkdtempSync(join(tmpdir(), "rayvan-projects-"));
    const dbPath = join(directory, "rayvan.db");

    try {
      const firstConnection = new LocalDatabaseConnection(dbPath);
      const firstRepository = new SqliteProjectRepository(firstConnection);
      const created = await firstRepository.create({
        name: "Persistent",
        description: "Survives restart",
      });
      firstConnection.close();

      const secondConnection = new LocalDatabaseConnection(dbPath);
      const secondRepository = new SqliteProjectRepository(secondConnection);
      const loaded = await secondRepository.getById(created.id);
      secondConnection.close();

      expect(loaded).toEqual(created);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });
});
