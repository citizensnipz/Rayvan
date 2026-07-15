import type { PluginExecutionActor } from "@rayvan/plugin-sdk";

import { PluginDomainError } from "../errors.js";
import type { EnvironmentMappingSuggestionRecord } from "../models.js";
import type { EnvironmentMappingSuggestionRepository } from "../repositories/types.js";

export class EnvironmentMappingService {
  constructor(
    private readonly suggestions: EnvironmentMappingSuggestionRepository,
  ) {}

  async createSuggestion(input: {
    projectId: string;
    connectionId: string;
    discoveredResourceId: string;
    suggestedEnvironmentId?: string;
    suggestedEnvironmentName?: string;
    confidence?: number;
    reasons: string[];
  }): Promise<EnvironmentMappingSuggestionRecord> {
    const record: EnvironmentMappingSuggestionRecord = {
      id: crypto.randomUUID(),
      projectId: input.projectId,
      connectionId: input.connectionId,
      discoveredResourceId: input.discoveredResourceId,
      suggestedEnvironmentId: input.suggestedEnvironmentId,
      suggestedEnvironmentName: input.suggestedEnvironmentName,
      confidence: input.confidence,
      reasons: [...input.reasons],
      status: "pending",
      createdAt: new Date().toISOString(),
    };
    await this.suggestions.save(record);
    return record;
  }

  async accept(input: {
    suggestionId: string;
    resolvedBy: PluginExecutionActor;
  }): Promise<EnvironmentMappingSuggestionRecord> {
    return this.resolve(input.suggestionId, "accepted", input.resolvedBy);
  }

  async reject(input: {
    suggestionId: string;
    resolvedBy: PluginExecutionActor;
  }): Promise<EnvironmentMappingSuggestionRecord> {
    return this.resolve(input.suggestionId, "rejected", input.resolvedBy);
  }

  async listPending(
    projectId: string,
  ): Promise<EnvironmentMappingSuggestionRecord[]> {
    return this.suggestions.listPendingByProjectId(projectId);
  }

  private async resolve(
    suggestionId: string,
    status: "accepted" | "rejected",
    resolvedBy: PluginExecutionActor,
  ): Promise<EnvironmentMappingSuggestionRecord> {
    const existing = await this.suggestions.getById(suggestionId);
    if (!existing) {
      throw new PluginDomainError(
        `Mapping suggestion not found: ${suggestionId}`,
      );
    }
    if (existing.status !== "pending") {
      throw new PluginDomainError(
        `Mapping suggestion is not pending: ${suggestionId}`,
      );
    }
    const updated: EnvironmentMappingSuggestionRecord = {
      ...existing,
      status,
      resolvedAt: new Date().toISOString(),
      resolvedBy,
    };
    await this.suggestions.save(updated);
    return updated;
  }
}
