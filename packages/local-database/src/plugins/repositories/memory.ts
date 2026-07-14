import type {
  ChangeApplyRecord,
  ChangePlanApprovalRecord,
  ChangePlanRecord,
  ChangePlanRejectionRecord,
  ChangePlanStatus,
  ChangeVerificationRecord,
  CredentialReferenceRecord,
  DesiredResourceStateRecord,
  DiscoveredResourceRecord,
  EnvironmentMappingSuggestionRecord,
  InstalledPluginRecord,
  ObservedResourceStateHistoryRecord,
  ObservedResourceStateRecord,
  PluginConnectionRecord,
  PluginExecutionHistoryRecord,
  PluginPermissionGrantRecord,
  ResourceBindingRecord,
} from "../models.js";
import { OptimisticConcurrencyError, PluginPersistenceError } from "../errors.js";
import type {
  ChangeApplyRepository,
  ChangePlanApprovalRepository,
  ChangePlanRepository,
  ChangeVerificationRepository,
  CredentialReferenceRepository,
  DesiredResourceStateRepository,
  DiscoveredResourceRepository,
  DiscoverySyncItem,
  EnvironmentMappingSuggestionRepository,
  InstalledPluginRepository,
  ObservedResourceStateRepository,
  PluginConnectionRepository,
  PluginExecutionHistoryRepository,
  PluginPermissionGrantRepository,
  ResourceBindingRepository,
} from "./types.js";

function clone<T>(value: T): T {
  return structuredClone(value);
}

export class InMemoryInstalledPluginRepository
  implements InstalledPluginRepository
{
  private readonly byId = new Map<string, InstalledPluginRecord>();

  async save(record: InstalledPluginRecord): Promise<void> {
    this.byId.set(record.id, clone(record));
  }

  async getById(id: string): Promise<InstalledPluginRecord | undefined> {
    const record = this.byId.get(id);
    return record ? clone(record) : undefined;
  }

  async getByPluginId(
    pluginId: string,
  ): Promise<InstalledPluginRecord | undefined> {
    for (const record of this.byId.values()) {
      if (record.pluginId === pluginId) {
        return clone(record);
      }
    }
    return undefined;
  }

  async list(): Promise<InstalledPluginRecord[]> {
    return [...this.byId.values()].map(clone);
  }

  async setEnabled(id: string, enabled: boolean): Promise<void> {
    const existing = this.byId.get(id);
    if (!existing) {
      throw new PluginPersistenceError(`Installed plugin not found: ${id}`);
    }
    this.byId.set(id, {
      ...existing,
      enabled,
      status: enabled ? "installed" : "disabled",
      updatedAt: new Date().toISOString(),
    });
  }

  async updateStatus(
    id: string,
    status: InstalledPluginRecord["status"],
  ): Promise<void> {
    const existing = this.byId.get(id);
    if (!existing) {
      throw new PluginPersistenceError(`Installed plugin not found: ${id}`);
    }
    this.byId.set(id, {
      ...existing,
      status,
      enabled: status === "installed" ? existing.enabled : false,
      updatedAt: new Date().toISOString(),
    });
  }
}

export class InMemoryPluginConnectionRepository
  implements PluginConnectionRepository
{
  private readonly byId = new Map<string, PluginConnectionRecord>();

  async save(record: PluginConnectionRecord): Promise<void> {
    this.byId.set(record.id, clone(record));
  }

  async getById(id: string): Promise<PluginConnectionRecord | undefined> {
    const record = this.byId.get(id);
    return record ? clone(record) : undefined;
  }

  async listByPluginId(pluginId: string): Promise<PluginConnectionRecord[]> {
    return [...this.byId.values()]
      .filter((record) => record.pluginId === pluginId)
      .map(clone);
  }

  async listByProjectId(projectId: string): Promise<PluginConnectionRecord[]> {
    return [...this.byId.values()]
      .filter((record) => record.projectId === projectId)
      .map(clone);
  }

  async listByInstalledPluginId(
    installedPluginId: string,
  ): Promise<PluginConnectionRecord[]> {
    return [...this.byId.values()]
      .filter((record) => record.installedPluginId === installedPluginId)
      .map(clone);
  }
}

export class InMemoryCredentialReferenceRepository
  implements CredentialReferenceRepository
{
  private readonly byId = new Map<string, CredentialReferenceRecord>();

  async save(record: CredentialReferenceRecord): Promise<void> {
    this.byId.set(record.id, clone(record));
  }

  async getById(id: string): Promise<CredentialReferenceRecord | undefined> {
    const record = this.byId.get(id);
    return record ? clone(record) : undefined;
  }

  async listByConnectionId(
    connectionId: string,
  ): Promise<CredentialReferenceRecord[]> {
    return [...this.byId.values()]
      .filter((record) => record.connectionId === connectionId)
      .map(clone);
  }
}

export class InMemoryPluginPermissionGrantRepository
  implements PluginPermissionGrantRepository
{
  private readonly byId = new Map<string, PluginPermissionGrantRecord>();

  async save(record: PluginPermissionGrantRecord): Promise<void> {
    this.byId.set(record.id, clone(record));
  }

  async getById(
    id: string,
  ): Promise<PluginPermissionGrantRecord | undefined> {
    const record = this.byId.get(id);
    return record ? clone(record) : undefined;
  }

  async listByConnectionId(
    connectionId: string,
  ): Promise<PluginPermissionGrantRecord[]> {
    return [...this.byId.values()]
      .filter((record) => record.connectionId === connectionId)
      .map(clone);
  }

  async listActiveByConnectionId(
    connectionId: string,
  ): Promise<PluginPermissionGrantRecord[]> {
    return [...this.byId.values()]
      .filter(
        (record) =>
          record.connectionId === connectionId &&
          record.granted &&
          !record.revokedAt,
      )
      .map(clone);
  }

  async replaceActiveGrants(input: {
    connectionId: string;
    projectId?: string;
    environmentId?: string;
    grants: PluginPermissionGrantRecord[];
  }): Promise<void> {
    const now = new Date().toISOString();
    for (const [id, record] of this.byId.entries()) {
      if (
        record.connectionId === input.connectionId &&
        record.granted &&
        !record.revokedAt &&
        record.projectId === input.projectId &&
        record.environmentId === input.environmentId
      ) {
        this.byId.set(id, {
          ...record,
          granted: false,
          revokedAt: now,
          reason: record.reason ?? "replaced",
        });
      }
    }
    for (const grant of input.grants) {
      this.byId.set(grant.id, clone(grant));
    }
  }
}

export class InMemoryDiscoveredResourceRepository
  implements DiscoveredResourceRepository
{
  private readonly byId = new Map<string, DiscoveredResourceRecord>();

  async save(record: DiscoveredResourceRecord): Promise<void> {
    const key = `${record.connectionId}|${record.providerResourceId}|${record.resourceType}`;
    for (const existing of this.byId.values()) {
      const existingKey = `${existing.connectionId}|${existing.providerResourceId}|${existing.resourceType}`;
      if (existingKey === key && existing.id !== record.id) {
        throw new PluginPersistenceError(
          `Duplicate discovered resource for connection ${record.connectionId}`,
        );
      }
    }
    this.byId.set(record.id, clone(record));
  }

  async getById(id: string): Promise<DiscoveredResourceRecord | undefined> {
    const record = this.byId.get(id);
    return record ? clone(record) : undefined;
  }

  async getByProviderKey(input: {
    connectionId: string;
    providerResourceId: string;
    resourceType: string;
  }): Promise<DiscoveredResourceRecord | undefined> {
    for (const record of this.byId.values()) {
      if (
        record.connectionId === input.connectionId &&
        record.providerResourceId === input.providerResourceId &&
        record.resourceType === input.resourceType
      ) {
        return clone(record);
      }
    }
    return undefined;
  }

  async listByConnectionId(
    connectionId: string,
  ): Promise<DiscoveredResourceRecord[]> {
    return [...this.byId.values()]
      .filter((record) => record.connectionId === connectionId)
      .map(clone);
  }

  async syncDiscovery(input: {
    pluginId: string;
    installedPluginId: string;
    connectionId: string;
    discoveredAt: string;
    items: DiscoverySyncItem[];
  }): Promise<DiscoveredResourceRecord[]> {
    const seenKeys = new Set<string>();
    const results: DiscoveredResourceRecord[] = [];

    for (const item of input.items) {
      const key = `${item.providerResourceId}|${item.resourceType}`;
      if (seenKeys.has(key)) {
        throw new PluginPersistenceError(
          `Duplicate provider resource in discovery batch: ${key}`,
        );
      }
      seenKeys.add(key);

      const existing = await this.getByProviderKey({
        connectionId: input.connectionId,
        providerResourceId: item.providerResourceId,
        resourceType: item.resourceType,
      });

      if (existing) {
        const updated: DiscoveredResourceRecord = {
          ...existing,
          name: item.name,
          parentProviderResourceId: item.parentProviderResourceId,
          metadata: item.metadata,
          pluginVersion: item.pluginVersion,
          schemaVersion: item.schemaVersion,
          discoveryStatus: "active",
          lastDiscoveredAt: input.discoveredAt,
          missingSince: undefined,
        };
        this.byId.set(updated.id, updated);
        results.push(clone(updated));
      } else {
        const created: DiscoveredResourceRecord = {
          id: crypto.randomUUID(),
          pluginId: input.pluginId,
          installedPluginId: input.installedPluginId,
          connectionId: input.connectionId,
          providerResourceId: item.providerResourceId,
          resourceType: item.resourceType,
          name: item.name,
          parentProviderResourceId: item.parentProviderResourceId,
          metadata: item.metadata,
          pluginVersion: item.pluginVersion,
          schemaVersion: item.schemaVersion,
          discoveryStatus: "active",
          firstDiscoveredAt: input.discoveredAt,
          lastDiscoveredAt: input.discoveredAt,
        };
        this.byId.set(created.id, created);
        results.push(clone(created));
      }
    }

    for (const record of this.byId.values()) {
      if (record.connectionId !== input.connectionId) {
        continue;
      }
      const key = `${record.providerResourceId}|${record.resourceType}`;
      if (!seenKeys.has(key) && record.discoveryStatus === "active") {
        this.byId.set(record.id, {
          ...record,
          discoveryStatus: "missing",
          missingSince: input.discoveredAt,
        });
      }
    }

    return results;
  }

  async markStatus(
    id: string,
    status: DiscoveredResourceRecord["discoveryStatus"],
    missingSince?: string,
  ): Promise<void> {
    const existing = this.byId.get(id);
    if (!existing) {
      throw new PluginPersistenceError(`Discovered resource not found: ${id}`);
    }
    this.byId.set(id, {
      ...existing,
      discoveryStatus: status,
      missingSince:
        status === "missing" ? (missingSince ?? new Date().toISOString()) : undefined,
    });
  }
}

export class InMemoryResourceBindingRepository
  implements ResourceBindingRepository
{
  private readonly byId = new Map<string, ResourceBindingRecord>();

  async save(record: ResourceBindingRecord): Promise<void> {
    this.byId.set(record.id, clone(record));
  }

  async getById(id: string): Promise<ResourceBindingRecord | undefined> {
    const record = this.byId.get(id);
    return record ? clone(record) : undefined;
  }

  async listByProjectId(projectId: string): Promise<ResourceBindingRecord[]> {
    return [...this.byId.values()]
      .filter((record) => record.projectId === projectId)
      .map(clone);
  }

  async listByConnectionId(
    connectionId: string,
  ): Promise<ResourceBindingRecord[]> {
    return [...this.byId.values()]
      .filter((record) => record.connectionId === connectionId)
      .map(clone);
  }

  async listByDiscoveredResourceId(
    discoveredResourceId: string,
  ): Promise<ResourceBindingRecord[]> {
    return [...this.byId.values()]
      .filter((record) => record.discoveredResourceId === discoveredResourceId)
      .map(clone);
  }

  async invalidateByConnectionId(connectionId: string): Promise<void> {
    const now = new Date().toISOString();
    for (const [id, record] of this.byId.entries()) {
      if (record.connectionId === connectionId && record.bindingStatus === "active") {
        this.byId.set(id, {
          ...record,
          bindingStatus: "invalid",
          updatedAt: now,
        });
      }
    }
  }
}

export class InMemoryEnvironmentMappingSuggestionRepository
  implements EnvironmentMappingSuggestionRepository
{
  private readonly byId = new Map<string, EnvironmentMappingSuggestionRecord>();

  async save(record: EnvironmentMappingSuggestionRecord): Promise<void> {
    this.byId.set(record.id, clone(record));
  }

  async getById(
    id: string,
  ): Promise<EnvironmentMappingSuggestionRecord | undefined> {
    const record = this.byId.get(id);
    return record ? clone(record) : undefined;
  }

  async listByProjectId(
    projectId: string,
  ): Promise<EnvironmentMappingSuggestionRecord[]> {
    return [...this.byId.values()]
      .filter((record) => record.projectId === projectId)
      .map(clone);
  }

  async listPendingByProjectId(
    projectId: string,
  ): Promise<EnvironmentMappingSuggestionRecord[]> {
    return [...this.byId.values()]
      .filter(
        (record) =>
          record.projectId === projectId && record.status === "pending",
      )
      .map(clone);
  }
}

export class InMemoryObservedResourceStateRepository
  implements ObservedResourceStateRepository
{
  private readonly latest = new Map<string, ObservedResourceStateRecord>();
  private readonly history: ObservedResourceStateHistoryRecord[] = [];

  async upsertLatest(
    record: ObservedResourceStateRecord,
  ): Promise<ObservedResourceStateRecord> {
    const existing = this.latest.get(record.discoveredResourceId);
    const next = clone({
      ...record,
      id: existing?.id ?? record.id,
    });
    this.latest.set(record.discoveredResourceId, next);
    this.history.push(clone({ ...next, id: crypto.randomUUID() }));
    return clone(next);
  }

  async getLatestByDiscoveredResourceId(
    discoveredResourceId: string,
  ): Promise<ObservedResourceStateRecord | undefined> {
    const record = this.latest.get(discoveredResourceId);
    return record ? clone(record) : undefined;
  }

  async listHistory(
    discoveredResourceId: string,
  ): Promise<ObservedResourceStateHistoryRecord[]> {
    return this.history
      .filter((record) => record.discoveredResourceId === discoveredResourceId)
      .map(clone);
  }
}

export class InMemoryDesiredResourceStateRepository
  implements DesiredResourceStateRepository
{
  private readonly byBindingId = new Map<string, DesiredResourceStateRecord>();

  async getByBindingId(
    resourceBindingId: string,
  ): Promise<DesiredResourceStateRecord | undefined> {
    const record = this.byBindingId.get(resourceBindingId);
    return record ? clone(record) : undefined;
  }

  async saveNew(
    record: DesiredResourceStateRecord,
  ): Promise<DesiredResourceStateRecord> {
    if (this.byBindingId.has(record.resourceBindingId)) {
      throw new PluginPersistenceError(
        `Desired state already exists for binding ${record.resourceBindingId}`,
      );
    }
    this.byBindingId.set(record.resourceBindingId, clone(record));
    return clone(record);
  }

  async updateWithExpectedRevision(input: {
    resourceBindingId: string;
    expectedRevision: number;
    next: DesiredResourceStateRecord;
  }): Promise<DesiredResourceStateRecord> {
    const existing = this.byBindingId.get(input.resourceBindingId);
    if (!existing) {
      throw new PluginPersistenceError(
        `Desired state not found for binding ${input.resourceBindingId}`,
      );
    }
    if (existing.revision !== input.expectedRevision) {
      throw new OptimisticConcurrencyError(
        `Desired state revision mismatch: expected ${input.expectedRevision}, found ${existing.revision}`,
      );
    }
    this.byBindingId.set(input.resourceBindingId, clone(input.next));
    return clone(input.next);
  }
}

export class InMemoryChangePlanRepository implements ChangePlanRepository {
  private readonly byId = new Map<string, ChangePlanRecord>();

  async save(record: ChangePlanRecord): Promise<void> {
    if (this.byId.has(record.id)) {
      throw new PluginPersistenceError(
        `Change plan is immutable after creation: ${record.id}`,
      );
    }
    this.byId.set(record.id, clone(record));
  }

  async getById(id: string): Promise<ChangePlanRecord | undefined> {
    const record = this.byId.get(id);
    return record ? clone(record) : undefined;
  }

  async listByBindingId(
    resourceBindingId: string,
  ): Promise<ChangePlanRecord[]> {
    return [...this.byId.values()]
      .filter((record) => record.resourceBindingId === resourceBindingId)
      .map(clone);
  }

  async setStatus(id: string, status: ChangePlanStatus): Promise<void> {
    const existing = this.byId.get(id);
    if (!existing) {
      throw new PluginPersistenceError(`Change plan not found: ${id}`);
    }
    this.byId.set(id, { ...existing, status });
  }

  async supersede(input: {
    planId: string;
    supersededByPlanId: string;
  }): Promise<void> {
    const existing = this.byId.get(input.planId);
    if (!existing) {
      throw new PluginPersistenceError(`Change plan not found: ${input.planId}`);
    }
    const supersedable: ChangePlanStatus[] = [
      "pending",
      "approved",
      "failed",
      "rejected",
    ];
    if (!supersedable.includes(existing.status)) {
      throw new PluginPersistenceError(
        `Cannot supersede change plan in status ${existing.status}`,
      );
    }
    this.byId.set(input.planId, {
      ...existing,
      status: "superseded",
      supersededByPlanId: input.supersededByPlanId,
    });
  }
}

export class InMemoryChangePlanApprovalRepository
  implements ChangePlanApprovalRepository
{
  private readonly approvals: ChangePlanApprovalRecord[] = [];
  private readonly rejections: ChangePlanRejectionRecord[] = [];

  constructor(private readonly plans: InMemoryChangePlanRepository) {}

  async appendApproval(record: ChangePlanApprovalRecord): Promise<void> {
    this.approvals.push(clone(record));
  }

  async appendRejection(record: ChangePlanRejectionRecord): Promise<void> {
    this.rejections.push(clone(record));
  }

  async listApprovalsByPlanId(
    changePlanId: string,
  ): Promise<ChangePlanApprovalRecord[]> {
    return this.approvals
      .filter((record) => record.changePlanId === changePlanId)
      .map(clone);
  }

  async listRejectionsByPlanId(
    changePlanId: string,
  ): Promise<ChangePlanRejectionRecord[]> {
    return this.rejections
      .filter((record) => record.changePlanId === changePlanId)
      .map(clone);
  }

  async getLatestApproval(
    changePlanId: string,
  ): Promise<ChangePlanApprovalRecord | undefined> {
    const matches = this.approvals.filter(
      (record) => record.changePlanId === changePlanId,
    );
    const latest = matches[matches.length - 1];
    return latest ? clone(latest) : undefined;
  }

  async approveAndTransitionPlan(input: {
    approval: ChangePlanApprovalRecord;
    planId: string;
  }): Promise<void> {
    this.approvals.push(clone(input.approval));
    await this.plans.setStatus(input.planId, "approved");
  }

  async rejectAndTransitionPlan(input: {
    rejection: ChangePlanRejectionRecord;
    planId: string;
  }): Promise<void> {
    this.rejections.push(clone(input.rejection));
    await this.plans.setStatus(input.planId, "rejected");
  }
}

export class InMemoryChangeApplyRepository implements ChangeApplyRepository {
  private readonly byId = new Map<string, ChangeApplyRecord>();

  constructor(private readonly plans: InMemoryChangePlanRepository) {}

  async save(record: ChangeApplyRecord): Promise<void> {
    this.byId.set(record.id, clone(record));
  }

  async getById(id: string): Promise<ChangeApplyRecord | undefined> {
    const record = this.byId.get(id);
    return record ? clone(record) : undefined;
  }

  async listByPlanId(changePlanId: string): Promise<ChangeApplyRecord[]> {
    return [...this.byId.values()]
      .filter((record) => record.changePlanId === changePlanId)
      .map(clone);
  }

  async beginApply(planId: string): Promise<void> {
    await this.plans.setStatus(planId, "applying");
  }

  async completeApply(input: {
    planId: string;
    apply: ChangeApplyRecord;
    planStatus: Extract<ChangePlanStatus, "applied" | "failed">;
  }): Promise<void> {
    this.byId.set(input.apply.id, clone(input.apply));
    await this.plans.setStatus(input.planId, input.planStatus);
  }
}

export class InMemoryChangeVerificationRepository
  implements ChangeVerificationRepository
{
  private readonly byId = new Map<string, ChangeVerificationRecord>();

  async save(record: ChangeVerificationRecord): Promise<void> {
    this.byId.set(record.id, clone(record));
  }

  async getById(id: string): Promise<ChangeVerificationRecord | undefined> {
    const record = this.byId.get(id);
    return record ? clone(record) : undefined;
  }

  async listByApplyId(
    changeApplyId: string,
  ): Promise<ChangeVerificationRecord[]> {
    return [...this.byId.values()]
      .filter((record) => record.changeApplyId === changeApplyId)
      .map(clone);
  }
}

export class InMemoryPluginExecutionHistoryRepository
  implements PluginExecutionHistoryRepository
{
  private readonly byId = new Map<string, PluginExecutionHistoryRecord>();

  async append(record: PluginExecutionHistoryRecord): Promise<void> {
    this.byId.set(record.id, clone(record));
  }

  async getByExecutionId(
    executionId: string,
  ): Promise<PluginExecutionHistoryRecord | undefined> {
    for (const record of this.byId.values()) {
      if (record.executionId === executionId) {
        return clone(record);
      }
    }
    return undefined;
  }

  async listByPluginId(
    pluginId: string,
  ): Promise<PluginExecutionHistoryRecord[]> {
    return [...this.byId.values()]
      .filter((record) => record.pluginId === pluginId)
      .map(clone);
  }

  async listByConnectionId(
    connectionId: string,
  ): Promise<PluginExecutionHistoryRecord[]> {
    return [...this.byId.values()]
      .filter((record) => record.connectionId === connectionId)
      .map(clone);
  }
}

export interface InMemoryPluginPersistence {
  installedPlugins: InMemoryInstalledPluginRepository;
  connections: InMemoryPluginConnectionRepository;
  credentialReferences: InMemoryCredentialReferenceRepository;
  permissionGrants: InMemoryPluginPermissionGrantRepository;
  discoveredResources: InMemoryDiscoveredResourceRepository;
  resourceBindings: InMemoryResourceBindingRepository;
  mappingSuggestions: InMemoryEnvironmentMappingSuggestionRepository;
  observedState: InMemoryObservedResourceStateRepository;
  desiredState: InMemoryDesiredResourceStateRepository;
  changePlans: InMemoryChangePlanRepository;
  changePlanApprovals: InMemoryChangePlanApprovalRepository;
  changeApplies: InMemoryChangeApplyRepository;
  changeVerifications: InMemoryChangeVerificationRepository;
  executionHistory: InMemoryPluginExecutionHistoryRepository;
}

export function createInMemoryPluginPersistence(): InMemoryPluginPersistence {
  const changePlans = new InMemoryChangePlanRepository();
  return {
    installedPlugins: new InMemoryInstalledPluginRepository(),
    connections: new InMemoryPluginConnectionRepository(),
    credentialReferences: new InMemoryCredentialReferenceRepository(),
    permissionGrants: new InMemoryPluginPermissionGrantRepository(),
    discoveredResources: new InMemoryDiscoveredResourceRepository(),
    resourceBindings: new InMemoryResourceBindingRepository(),
    mappingSuggestions: new InMemoryEnvironmentMappingSuggestionRepository(),
    observedState: new InMemoryObservedResourceStateRepository(),
    desiredState: new InMemoryDesiredResourceStateRepository(),
    changePlans,
    changePlanApprovals: new InMemoryChangePlanApprovalRepository(changePlans),
    changeApplies: new InMemoryChangeApplyRepository(changePlans),
    changeVerifications: new InMemoryChangeVerificationRepository(),
    executionHistory: new InMemoryPluginExecutionHistoryRepository(),
  };
}
