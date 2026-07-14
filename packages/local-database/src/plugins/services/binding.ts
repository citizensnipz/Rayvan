import type { PluginExecutionActor } from "@rayvan/plugin-sdk";

import { PluginDomainError } from "../errors.js";
import type { ResourceBindingRecord, ResourceBindingStatus } from "../models.js";
import type {
  DiscoveredResourceRepository,
  ResourceBindingRepository,
} from "../repositories/types.js";

export interface CreateBindingInput {
  projectId: string;
  /** When set, caller must supply expectedProjectIdForEnvironment matching projectId. */
  environmentId?: string;
  expectedProjectIdForEnvironment?: string;
  discoveredResourceId: string;
  role?: string;
  displayName?: string;
  bindingStatus?: ResourceBindingStatus;
  createdBy: PluginExecutionActor;
}

export class ResourceBindingService {
  constructor(
    private readonly discoveredResources: DiscoveredResourceRepository,
    private readonly bindings: ResourceBindingRepository,
  ) {}

  async bind(input: CreateBindingInput): Promise<ResourceBindingRecord> {
    if (
      input.environmentId &&
      input.expectedProjectIdForEnvironment !== undefined &&
      input.expectedProjectIdForEnvironment !== input.projectId
    ) {
      throw new PluginDomainError(
        "Environment does not belong to the binding project",
      );
    }

    const resource = await this.discoveredResources.getById(
      input.discoveredResourceId,
    );
    if (!resource) {
      throw new PluginDomainError(
        `Discovered resource not found: ${input.discoveredResourceId}`,
      );
    }

    const now = new Date().toISOString();
    const record: ResourceBindingRecord = {
      id: crypto.randomUUID(),
      projectId: input.projectId,
      environmentId: input.environmentId,
      discoveredResourceId: resource.id,
      pluginId: resource.pluginId,
      connectionId: resource.connectionId,
      role: input.role,
      displayName: input.displayName,
      bindingStatus: input.bindingStatus ?? "active",
      createdBy: input.createdBy,
      createdAt: now,
      updatedAt: now,
    };
    await this.bindings.save(record);
    return record;
  }

  async detach(bindingId: string): Promise<ResourceBindingRecord> {
    const existing = await this.bindings.getById(bindingId);
    if (!existing) {
      throw new PluginDomainError(`Resource binding not found: ${bindingId}`);
    }
    const updated: ResourceBindingRecord = {
      ...existing,
      bindingStatus: "detached",
      updatedAt: new Date().toISOString(),
    };
    await this.bindings.save(updated);
    return updated;
  }

  async listByProjectId(projectId: string): Promise<ResourceBindingRecord[]> {
    return this.bindings.listByProjectId(projectId);
  }
}
