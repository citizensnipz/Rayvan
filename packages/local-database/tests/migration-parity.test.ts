import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { V3_PLUGIN_PERSISTENCE_SQL } from "../src/database/migrations.js";

describe("V3 plugin persistence SQL parity", () => {
  it("matches the shared Rust migration SQL file byte-for-byte", () => {
    const here = dirname(fileURLToPath(import.meta.url));
    const rustSqlPath = join(
      here,
      "../../../crates/local-store/migrations/v3_plugin_persistence.sql",
    );
    const rustSql = readFileSync(rustSqlPath, "utf8").replace(/\r\n/g, "\n");
    expect(V3_PLUGIN_PERSISTENCE_SQL.replace(/\r\n/g, "\n")).toBe(rustSql);
  });

  it("Rust migrations.rs includes the shared SQL file", () => {
    const here = dirname(fileURLToPath(import.meta.url));
    const rustMigrations = readFileSync(
      join(here, "../../../crates/local-store/src/migrations.rs"),
      "utf8",
    );
    expect(rustMigrations).toContain(
      'include_str!("../migrations/v3_plugin_persistence.sql")',
    );
    expect(rustMigrations).toContain("CURRENT_SCHEMA_VERSION: u32 = 3");
  });
});
