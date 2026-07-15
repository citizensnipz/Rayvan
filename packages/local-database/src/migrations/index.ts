import type { DatabaseSchemaVersion } from "../schema/index.js";
import {
  APP_SETTINGS_TABLE_MIGRATION,
  MIGRATION_VERSION,
  PROJECTS_TABLE_MIGRATION,
  V3_PLUGIN_PERSISTENCE_SQL,
  V4_ENVIRONMENTS_AND_CONFIGURATION_SQL,
  V5_DESIRED_APPLIED_CONFIGURATION_SQL,
} from "../database/migrations.js";

export interface Migration {
  version: number;
  up: string;
  down?: string;
}

export const INITIAL_MIGRATIONS: Migration[] = [
  {
    version: 1,
    up: PROJECTS_TABLE_MIGRATION,
  },
  {
    version: 2,
    up: APP_SETTINGS_TABLE_MIGRATION,
  },
  {
    version: 3,
    up: V3_PLUGIN_PERSISTENCE_SQL,
  },
  {
    version: 4,
    up: V4_ENVIRONMENTS_AND_CONFIGURATION_SQL,
  },
  {
    version: 5,
    up: V5_DESIRED_APPLIED_CONFIGURATION_SQL,
  },
];

export function listPendingMigrations(
  currentVersion: number,
  target: DatabaseSchemaVersion,
): Migration[] {
  return INITIAL_MIGRATIONS.filter(
    (migration) =>
      migration.version > currentVersion && migration.version <= target.version,
  );
}

export { MIGRATION_VERSION };
