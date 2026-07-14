import type { PluginExecutionActor } from "@rayvan/plugin-sdk";

import { PluginDomainError } from "../errors.js";
import type {
  DesiredResourceStateRecord,
  ObservedResourceStateRecord,
} from "../models.js";
import type {
  DesiredResourceStateRepository,
  ObservedResourceStateRepository,
} from "../repositories/types.js";
import { assertNoPlaintextSecrets } from "../secrets.js";

export class ResourceStateService {
  constructor(
    private readonly observed: ObservedResourceStateRepository,
    private readonly desired: DesiredResourceStateRepository,
  ) {}

  async recordObserved(
    input: Omit<ObservedResourceStateRecord, "id"> & { id?: string },
  ): Promise<ObservedResourceStateRecord> {
    assertNoPlaintextSecrets(input.state, "observed.state");
    return this.observed.upsertLatest({
      id: input.id ?? crypto.randomUUID(),
      ...input,
    });
  }

  async getObserved(
    discoveredResourceId: string,
  ): Promise<ObservedResourceStateRecord | undefined> {
    return this.observed.getLatestByDiscoveredResourceId(discoveredResourceId);
  }

  async listObservedHistory(
    discoveredResourceId: string,
  ): Promise<ObservedResourceStateRecord[]> {
    return this.observed.listHistory(discoveredResourceId);
  }

  async saveDesired(input: {
    projectId: string;
    environmentId?: string;
    resourceBindingId: string;
    pluginId: string;
    connectionId: string;
    state: Record<string, unknown>;
    schemaVersion: string;
    createdBy: PluginExecutionActor;
    expectedRevision?: number;
  }): Promise<DesiredResourceStateRecord> {
    assertNoPlaintextSecrets(input.state, "desired.state");

    const existing = await this.desired.getByBindingId(input.resourceBindingId);
    const now = new Date().toISOString();

    if (!existing) {
      const created: DesiredResourceStateRecord = {
        id: crypto.randomUUID(),
        projectId: input.projectId,
        environmentId: input.environmentId,
        resourceBindingId: input.resourceBindingId,
        pluginId: input.pluginId,
        connectionId: input.connectionId,
        state: structuredClone(input.state),
        schemaVersion: input.schemaVersion,
        revision: 1,
        createdBy: input.createdBy,
        createdAt: now,
        updatedAt: now,
      };
      return this.desired.saveNew(created);
    }

    if (input.expectedRevision === undefined) {
      throw new PluginDomainError(
        "expectedRevision is required when updating desired state",
      );
    }

    const next: DesiredResourceStateRecord = {
      ...existing,
      environmentId: input.environmentId,
      state: structuredClone(input.state),
      schemaVersion: input.schemaVersion,
      revision: existing.revision + 1,
      createdBy: input.createdBy,
      updatedAt: now,
    };

    return this.desired.updateWithExpectedRevision({
      resourceBindingId: input.resourceBindingId,
      expectedRevision: input.expectedRevision,
      next,
    });
  }

  async getDesired(
    resourceBindingId: string,
  ): Promise<DesiredResourceStateRecord | undefined> {
    return this.desired.getByBindingId(resourceBindingId);
  }
}
