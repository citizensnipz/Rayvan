import type {
  AppliedConfigurationState,
  ConfigurationKey,
  ConfigurationOccurrence,
  DesiredConfigurationValue,
  Environment,
  FindingEvaluationRunRecord,
  FindingLifecycleEventRecord,
  FindingRecord,
} from "@rayvan/core";
import {
  appliedConfigurationStateId,
  configurationKeyId,
  configurationOccurrenceId,
  desiredConfigurationValueId,
  environmentId,
  findingEvaluationRunId,
  findingId,
  projectId,
} from "@rayvan/core";

import { FindingEngine } from "../src/engine/finding-engine.js";
import type {
  FindingEngineRepositories,
  FindingEvaluationRunRepository,
  FindingLifecycleEventRepository,
  FindingQuery,
  FindingRepository,
  ProjectFindingsContext,
} from "../src/types.js";

export const PROJECT = projectId("proj-1");
export const ENV_DEV = environmentId("env-dev");
export const NOW = "2026-07-16T12:00:00.000Z";

export function makeEnvironment(
  id = ENV_DEV,
  overrides: Partial<Environment> = {},
): Environment {
  return {
    id,
    projectId: PROJECT,
    name: overrides.name ?? "Development",
    slug: overrides.slug ?? "dev",
    kind: overrides.kind ?? "development",
    status: overrides.status ?? "healthy",
    createdAt: overrides.createdAt ?? "2026-01-01T00:00:00.000Z",
    updatedAt: overrides.updatedAt ?? "2026-01-01T00:00:00.000Z",
    ...overrides,
  };
}

export function makeKey(
  overrides: Partial<ConfigurationKey> & { name: string },
): ConfigurationKey {
  return {
    id: overrides.id ?? configurationKeyId(`key-${overrides.name}`),
    projectId: PROJECT,
    name: overrides.name,
    valueType: overrides.valueType ?? "string",
    required: overrides.required ?? false,
    sensitive: overrides.sensitive ?? false,
    source: overrides.source ?? "manual",
    createdAt: NOW,
    updatedAt: NOW,
    ...overrides,
  };
}

export function makeDesired(
  keyId: string,
  environment = ENV_DEV,
  overrides: Partial<DesiredConfigurationValue> = {},
): DesiredConfigurationValue {
  return {
    id: desiredConfigurationValueId(`desired-${keyId}`),
    configurationKeyId: configurationKeyId(keyId),
    environmentId: environment,
    projectId: PROJECT,
    desiredValue: overrides.desiredValue ?? "expected",
    valueFingerprint: overrides.valueFingerprint ?? "fp-expected",
    revision: overrides.revision ?? 1,
    createdAt: NOW,
    updatedAt: NOW,
    updatedBy: { kind: "user", id: "u1" },
    ...overrides,
  };
}

export function makeOccurrence(
  overrides: Partial<ConfigurationOccurrence> & {
    configurationKeyId: string;
  },
): ConfigurationOccurrence {
  return {
    id: configurationOccurrenceId(overrides.id ?? crypto.randomUUID()),
    configurationKeyId: configurationKeyId(overrides.configurationKeyId),
    projectId: PROJECT,
    environmentId: overrides.environmentId ?? ENV_DEV,
    pluginId: overrides.pluginId ?? "plugin-a",
    connectionId: overrides.connectionId ?? "conn-1",
    discoveredResourceId: overrides.discoveredResourceId ?? "res-1",
    resourceBindingId: overrides.resourceBindingId,
    providerKey: overrides.providerKey ?? "KEY",
    valueAccess: overrides.valueAccess ?? "readable",
    observedValue: overrides.observedValue,
    valueFingerprint: overrides.valueFingerprint,
    firstObservedAt: overrides.firstObservedAt ?? NOW,
    lastObservedAt: overrides.lastObservedAt ?? NOW,
    ...overrides,
  };
}

export function makeApplied(
  overrides: Partial<AppliedConfigurationState> & {
    configurationKeyId: string;
  },
): AppliedConfigurationState {
  return {
    id: appliedConfigurationStateId(overrides.id ?? crypto.randomUUID()),
    configurationKeyId: configurationKeyId(overrides.configurationKeyId),
    environmentId: overrides.environmentId ?? ENV_DEV,
    projectId: PROJECT,
    resourceBindingId: overrides.resourceBindingId ?? "binding-1",
    desiredRevision: overrides.desiredRevision ?? 1,
    appliedFingerprint: overrides.appliedFingerprint ?? "fp-expected",
    applyExecutionId: overrides.applyExecutionId ?? "exec-1",
    status: overrides.status ?? "applied",
    appliedAt: overrides.appliedAt ?? NOW,
    ...overrides,
  };
}

export function emptyContext(
  overrides: Partial<ProjectFindingsContext> = {},
): ProjectFindingsContext {
  return {
    projectId: PROJECT,
    environments: [makeEnvironment()],
    keys: [],
    occurrences: [],
    desired: [],
    applied: [],
    connections: [],
    installedPlugins: [],
    discoveredResources: [],
    resourceBindings: [],
    mappingSuggestions: [],
    ...overrides,
  };
}

export class InMemoryFindingRepository implements FindingRepository {
  readonly byId = new Map<string, FindingRecord>();

  async getById(id: string): Promise<FindingRecord | undefined> {
    return this.byId.get(id);
  }

  async getByFingerprint(
    projectIdValue: string,
    fingerprint: string,
  ): Promise<FindingRecord | undefined> {
    return [...this.byId.values()].find(
      (record) =>
        String(record.projectId) === projectIdValue &&
        record.fingerprint === fingerprint,
    );
  }

  async list(query: FindingQuery): Promise<FindingRecord[]> {
    let records = [...this.byId.values()].filter(
      (record) => String(record.projectId) === String(query.projectId),
    );
    if (!query.includeResolved) {
      records = records.filter(
        (record) =>
          record.status === "open" ||
          record.status === "acknowledged" ||
          record.status === "suppressed",
      );
    }
    if (query.statuses) {
      records = records.filter((record) =>
        query.statuses!.includes(record.status),
      );
    }
    if (query.ruleId) {
      records = records.filter((record) => record.ruleId === query.ruleId);
    }
    if (query.environmentId) {
      records = records.filter(
        (record) => record.environmentId === query.environmentId,
      );
    }
    if (query.connectionId) {
      records = records.filter(
        (record) => record.connectionId === query.connectionId,
      );
    }
    if (query.limit !== undefined) {
      records = records.slice(0, query.limit);
    }
    return records;
  }

  async save(record: FindingRecord): Promise<void> {
    this.byId.set(record.id, structuredClone(record));
  }

  async saveMany(records: FindingRecord[]): Promise<void> {
    for (const record of records) {
      await this.save(record);
    }
  }

  seed(record: FindingRecord): void {
    this.byId.set(record.id, structuredClone(record));
  }
}

export class InMemoryLifecycleRepository
  implements FindingLifecycleEventRepository
{
  readonly events: FindingLifecycleEventRecord[] = [];

  async append(event: FindingLifecycleEventRecord): Promise<void> {
    this.events.push(event);
  }

  async listByFindingId(
    findingIdValue: string,
  ): Promise<FindingLifecycleEventRecord[]> {
    return this.events.filter(
      (event) => String(event.findingId) === findingIdValue,
    );
  }
}

export class InMemoryEvaluationRunRepository
  implements FindingEvaluationRunRepository
{
  readonly byId = new Map<string, FindingEvaluationRunRecord>();

  async save(run: FindingEvaluationRunRecord): Promise<void> {
    this.byId.set(run.id, run);
  }

  async getById(
    id: string,
  ): Promise<FindingEvaluationRunRecord | undefined> {
    return this.byId.get(id);
  }

  async listByProject(
    projectIdValue: string,
    limit?: number,
  ): Promise<FindingEvaluationRunRecord[]> {
    const runs = [...this.byId.values()].filter(
      (run) => String(run.projectId) === projectIdValue,
    );
    return limit === undefined ? runs : runs.slice(0, limit);
  }
}

export function createTestRepos(): FindingEngineRepositories & {
  findings: InMemoryFindingRepository;
  lifecycleEvents: InMemoryLifecycleRepository;
  evaluationRuns: InMemoryEvaluationRunRepository;
} {
  const findings = new InMemoryFindingRepository();
  const lifecycleEvents = new InMemoryLifecycleRepository();
  const evaluationRuns = new InMemoryEvaluationRunRepository();
  return { findings, lifecycleEvents, evaluationRuns };
}

export function createTestEngine(repos = createTestRepos()) {
  return {
    engine: new FindingEngine({ repositories: repos }),
    repos,
  };
}

export function makeFindingRecord(
  overrides: Partial<FindingRecord> & {
    ruleId: string;
    fingerprint: string;
  },
): FindingRecord {
  return {
    id: findingId(overrides.id ?? crypto.randomUUID()),
    projectId: PROJECT,
    ruleId: overrides.ruleId,
    source: overrides.source ?? { type: "rayvan" },
    category: overrides.category ?? "configuration",
    severity: overrides.severity ?? "warning",
    title: overrides.title ?? "Finding",
    summary: overrides.summary ?? "Summary",
    status: overrides.status ?? "open",
    fingerprint: overrides.fingerprint,
    fingerprintVersion: overrides.fingerprintVersion ?? "1",
    evidence: overrides.evidence ?? [],
    firstDetectedAt: overrides.firstDetectedAt ?? NOW,
    lastDetectedAt: overrides.lastDetectedAt ?? NOW,
    occurrenceCount: overrides.occurrenceCount ?? 1,
    lastEvaluationRunId:
      overrides.lastEvaluationRunId ??
      findingEvaluationRunId("run-seed"),
    metadata: overrides.metadata ?? {},
    schemaVersion: overrides.schemaVersion ?? "1",
    ...overrides,
  };
}
