import type { LocalDatabaseConnection } from "../../database/connection.js";
import {
  SqliteDiscoveredResourceRepository,
  SqliteInstalledPluginRepository,
  SqlitePluginConnectionRepository,
  SqlitePluginExecutionHistoryRepository,
  SqlitePluginPermissionGrantRepository,
} from "../sqlite/repositories.js";
import { SqlitePluginSetupSessionRepository } from "../sqlite/setup-sessions.js";
import {
  createInMemoryPluginPersistence,
  type InMemoryPluginPersistence,
} from "./memory.js";
import {
  InMemoryPluginSetupSessionRepository,
  type PluginSetupSessionRepository,
} from "./setup-sessions.js";
import type {
  ChangeApplyRepository,
  ChangePlanApprovalRepository,
  ChangePlanRepository,
  ChangeVerificationRepository,
  CredentialReferenceRepository,
  DesiredResourceStateRepository,
  DiscoveredResourceRepository,
  EnvironmentMappingSuggestionRepository,
  InstalledPluginRepository,
  ObservedResourceStateRepository,
  PluginConnectionRepository,
  PluginExecutionHistoryRepository,
  PluginPermissionGrantRepository,
  ResourceBindingRepository,
} from "./types.js";

/**
 * Persistence bundle used by the daemon plugin host.
 * Concrete repos may be SQLite or in-memory.
 */
export interface PluginPersistenceBundle {
  installedPlugins: InstalledPluginRepository;
  connections: PluginConnectionRepository;
  credentialReferences: CredentialReferenceRepository;
  permissionGrants: PluginPermissionGrantRepository;
  discoveredResources: DiscoveredResourceRepository;
  resourceBindings: ResourceBindingRepository;
  mappingSuggestions: EnvironmentMappingSuggestionRepository;
  observedState: ObservedResourceStateRepository;
  desiredState: DesiredResourceStateRepository;
  changePlans: ChangePlanRepository;
  changePlanApprovals: ChangePlanApprovalRepository;
  changeApplies: ChangeApplyRepository;
  changeVerifications: ChangeVerificationRepository;
  executionHistory: PluginExecutionHistoryRepository;
  setupSessions: PluginSetupSessionRepository;
}

/**
 * SQLite-backed installed plugins + connections (+ available sqlite adapters),
 * with in-memory fallbacks for entities that still lack sqlite repositories.
 */
export function createSqlitePluginPersistence(
  connection: LocalDatabaseConnection,
): PluginPersistenceBundle {
  const memory: InMemoryPluginPersistence = createInMemoryPluginPersistence();
  return {
    ...memory,
    installedPlugins: new SqliteInstalledPluginRepository(connection),
    connections: new SqlitePluginConnectionRepository(connection),
    permissionGrants: new SqlitePluginPermissionGrantRepository(connection),
    discoveredResources: new SqliteDiscoveredResourceRepository(connection),
    executionHistory: new SqlitePluginExecutionHistoryRepository(connection),
    setupSessions: new SqlitePluginSetupSessionRepository(connection),
  };
}

export function createInMemoryPluginPersistenceBundle(): PluginPersistenceBundle {
  return {
    ...createInMemoryPluginPersistence(),
    setupSessions: new InMemoryPluginSetupSessionRepository(),
  };
}
