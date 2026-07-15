/**
 * Sync the authoritative Rust v3 SQL file into the TypeScript migration const.
 *
 * Authority: crates/local-store/migrations/v3_plugin_persistence.sql
 * Mirror:    packages/local-database/src/database/migrations.ts
 *
 * Usage (from repo root):
 *   node packages/local-database/scripts/sync-v3-sql-from-rust.mjs
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "../../..");
const sqlPath = path.join(
  repoRoot,
  "crates/local-store/migrations/v3_plugin_persistence.sql",
);
const migrationsTsPath = path.join(
  repoRoot,
  "packages/local-database/src/database/migrations.ts",
);

const sql = fs.readFileSync(sqlPath, "utf8").replace(/\r\n/g, "\n");
const migrationsTs = fs.readFileSync(migrationsTsPath, "utf8");

const beginMarker =
  "export const V3_PLUGIN_PERSISTENCE_SQL = `\n";
const endMarker = "`;\n\nexport const MIGRATION_VERSION";

const begin = migrationsTs.indexOf(beginMarker);
const end = migrationsTs.indexOf(endMarker);
if (begin < 0 || end < 0 || end <= begin) {
  throw new Error(
    "Could not locate V3_PLUGIN_PERSISTENCE_SQL block in migrations.ts",
  );
}

const next =
  migrationsTs.slice(0, begin + beginMarker.length) +
  sql +
  migrationsTs.slice(end);

fs.writeFileSync(migrationsTsPath, next);
console.log(`Synced ${sql.length} chars from ${sqlPath} → migrations.ts`);
