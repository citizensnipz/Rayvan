import type { DatabaseSchemaVersion } from "../schema/index.js";

export interface Migration {
  version: number;
  up: string;
  down?: string;
}

export const INITIAL_MIGRATIONS: Migration[] = [
  {
    version: 1,
    up: "-- placeholder: create Rayvan metadata tables",
  },
];

export function listPendingMigrations(
  currentVersion: number,
  target: DatabaseSchemaVersion,
): Migration[] {
  return INITIAL_MIGRATIONS.filter(
    (migration) => migration.version > currentVersion && migration.version <= target.version,
  );
}
