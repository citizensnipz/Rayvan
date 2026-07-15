import {
  appliedConfigurationStateId,
  configurationKeyId,
  environmentId,
  projectId,
  type AppliedConfigurationState,
} from "@rayvan/core";

import type {
  AppliedConfigurationStateRepository,
  UpsertAppliedConfigurationStateInput,
} from "./applied-repository.js";

export class InMemoryAppliedConfigurationStateRepository
  implements AppliedConfigurationStateRepository
{
  private readonly states = new Map<string, AppliedConfigurationState>();

  async getById(id: string): Promise<AppliedConfigurationState | null> {
    return this.states.get(id) ?? null;
  }

  async getByKeyEnvironmentBinding(
    configurationKeyIdValue: string,
    environmentIdValue: string,
    resourceBindingId: string,
  ): Promise<AppliedConfigurationState | null> {
    return (
      [...this.states.values()].find(
        (state) =>
          state.configurationKeyId === configurationKeyIdValue &&
          state.environmentId === environmentIdValue &&
          state.resourceBindingId === resourceBindingId,
      ) ?? null
    );
  }

  async listByEnvironmentId(
    environmentIdValue: string,
  ): Promise<AppliedConfigurationState[]> {
    return [...this.states.values()]
      .filter((state) => state.environmentId === environmentIdValue)
      .sort((left, right) => right.appliedAt.localeCompare(left.appliedAt));
  }

  async listByProjectId(
    projectIdValue: string,
  ): Promise<AppliedConfigurationState[]> {
    return [...this.states.values()]
      .filter((state) => state.projectId === projectIdValue)
      .sort((left, right) => right.appliedAt.localeCompare(left.appliedAt));
  }

  async upsert(
    input: UpsertAppliedConfigurationStateInput,
  ): Promise<AppliedConfigurationState> {
    const existing = await this.getByKeyEnvironmentBinding(
      input.configurationKeyId,
      input.environmentId,
      input.resourceBindingId,
    );

    const state: AppliedConfigurationState = {
      id: existing?.id ?? appliedConfigurationStateId(crypto.randomUUID()),
      configurationKeyId: configurationKeyId(input.configurationKeyId),
      environmentId: environmentId(input.environmentId),
      projectId: projectId(input.projectId),
      resourceBindingId: input.resourceBindingId,
      desiredRevision: input.desiredRevision,
      appliedFingerprint: input.appliedFingerprint,
      applyExecutionId: input.applyExecutionId,
      verificationExecutionId: input.verificationExecutionId,
      status: input.status,
      appliedAt: input.appliedAt,
      verifiedAt: input.verifiedAt,
    };
    this.states.set(state.id, state);
    return state;
  }
}
