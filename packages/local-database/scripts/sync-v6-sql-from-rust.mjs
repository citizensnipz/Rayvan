/**
 * Sync the authoritative Rust v6 SQL file into the TypeScript migration const.
 *
 * Authority: crates/local-store/migrations/v6_findings.sql
 * Mirror:    packages/local-database/src/database/migrations.ts
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "../../..");
const sqlPath = path.join(
  repoRoot,
  "crates/local-store/migrations/v6_findings.sql",
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

const migrationsTs = fs.readFileSync(migrationsTsPath, "utf8").replace(/\r\n/g, "\n");
const beginMarker = "export const V6_FINDINGS_SQL = `";
const endMarker = "`;\n\nexport const MIGRATION_VERSION";

const begin = migrationsTs.indexOf(beginMarker);
const end = migrationsTs.indexOf(endMarker);
if (begin < 0 || end < 0 || end <= begin) {
  throw new Error("Could not locate V6_FINDINGS_SQL block in migrations.ts");
}

const next =
  migrationsTs.slice(0, begin + beginMarker.length) +
  sql +
  migrationsTs.slice(end);

fs.writeFileSync(migrationsTsPath, next.replace(/\n/g, "\r\n"));
console.log(`Synced ${sql.length} chars from ${sqlPath} → migrations.ts`);
