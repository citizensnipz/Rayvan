/**
 * Sync the authoritative Rust v7 SQL file into the TypeScript migration const.
 *
 * Authority: crates/local-store/migrations/v7_daemon_control_plane.sql
 * Mirror:    packages/local-database/src/database/migrations.ts
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "../../..");
const sqlPath = path.join(
  repoRoot,
  "crates/local-store/migrations/v7_daemon_control_plane.sql",
);
const migrationsTsPath = path.join(
  repoRoot,
  "packages/local-database/src/database/migrations.ts",
);

let sql = fs.readFileSync(sqlPath, "utf8").replace(/\r\n/g, "\n");
if (!sql.startsWith("\n")) {
  sql = `\n${sql}`;
}
if (!sql.endsWith("\n")) {
  sql = `${sql}\n`;
}
fs.writeFileSync(sqlPath, sql);

const migrationsTs = fs
  .readFileSync(migrationsTsPath, "utf8")
  .replace(/\r\n/g, "\n");

const beginMarker = "export const V7_DAEMON_CONTROL_PLANE_SQL = ";
const endMarker = "export const MIGRATION_VERSION = 7;";
const begin = migrationsTs.indexOf(beginMarker);
const end = migrationsTs.indexOf(endMarker);
if (begin < 0 || end < 0 || end <= begin) {
  throw new Error(
    "Could not locate V7_DAEMON_CONTROL_PLANE_SQL block in migrations.ts",
  );
}

const next =
  migrationsTs.slice(0, begin) +
  `export const V7_DAEMON_CONTROL_PLANE_SQL = \`${sql}\`;\n\n` +
  migrationsTs.slice(end);

fs.writeFileSync(migrationsTsPath, next);
console.log(`Synced ${sql.length} chars from ${sqlPath} → migrations.ts`);
