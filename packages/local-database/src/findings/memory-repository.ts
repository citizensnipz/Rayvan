import type { FindingRecord } from "@rayvan/core";
import { ACTIVE_FINDING_STATUSES } from "@rayvan/core";
import type {
  FindingQuery,
  FindingRepository,
} from "@rayvan/findings-engine";

function matchesSearch(record: FindingRecord, search: string): boolean {
  const needle = search.trim().toLowerCase();
  if (!needle) {
    return true;
  }
  return (
    record.title.toLowerCase().includes(needle) ||
    record.summary.toLowerCase().includes(needle) ||
    (record.description?.toLowerCase().includes(needle) ?? false) ||
    record.ruleId.toLowerCase().includes(needle) ||
    record.fingerprint.toLowerCase().includes(needle)
  );
}

export function filterFindingRecords(
  records: FindingRecord[],
  query: FindingQuery,
): FindingRecord[] {
  let filtered = records.filter(
    (record) => String(record.projectId) === String(query.projectId),
  );

  if (!query.includeResolved) {
    filtered = filtered.filter((record) =>
      (ACTIVE_FINDING_STATUSES as readonly string[]).includes(record.status),
    );
  }

  if (query.statuses?.length) {
    filtered = filtered.filter((record) =>
      query.statuses!.includes(record.status),
    );
  }
  if (query.severities?.length) {
    filtered = filtered.filter((record) =>
      query.severities!.includes(record.severity),
    );
  }
  if (query.categories?.length) {
    filtered = filtered.filter((record) =>
      query.categories!.includes(record.category),
    );
  }
  if (query.environmentId) {
    filtered = filtered.filter(
      (record) => record.environmentId === query.environmentId,
    );
  }
  if (query.connectionId) {
    filtered = filtered.filter(
      (record) => record.connectionId === query.connectionId,
    );
  }
  if (query.resourceBindingId) {
    filtered = filtered.filter(
      (record) => record.resourceBindingId === query.resourceBindingId,
    );
  }
  if (query.configurationKeyId) {
    filtered = filtered.filter(
      (record) => record.configurationKeyId === query.configurationKeyId,
    );
  }
  if (query.ruleId) {
    filtered = filtered.filter((record) => record.ruleId === query.ruleId);
  }
  if (query.search) {
    filtered = filtered.filter((record) => matchesSearch(record, query.search!));
  }

  filtered.sort((left, right) =>
    right.lastDetectedAt.localeCompare(left.lastDetectedAt),
  );

  if (query.limit !== undefined) {
    filtered = filtered.slice(0, query.limit);
  }

  return filtered;
}

export class InMemoryFindingRepository implements FindingRepository {
  readonly byId = new Map<string, FindingRecord>();

  async getById(id: string): Promise<FindingRecord | undefined> {
    const record = this.byId.get(id);
    return record ? structuredClone(record) : undefined;
  }

  async getByFingerprint(
    projectIdValue: string,
    fingerprint: string,
  ): Promise<FindingRecord | undefined> {
    const record = [...this.byId.values()].find(
      (item) =>
        String(item.projectId) === projectIdValue &&
        item.fingerprint === fingerprint,
    );
    return record ? structuredClone(record) : undefined;
  }

  async list(query: FindingQuery): Promise<FindingRecord[]> {
    return filterFindingRecords([...this.byId.values()], query).map((record) =>
      structuredClone(record),
    );
  }

  async save(record: FindingRecord): Promise<void> {
    this.byId.set(String(record.id), structuredClone(record));
  }

  async saveMany(records: FindingRecord[]): Promise<void> {
    for (const record of records) {
      await this.save(record);
    }
  }
}
