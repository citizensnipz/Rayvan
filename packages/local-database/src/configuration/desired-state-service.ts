import type {
  AppliedConfigurationState,
  AppliedConfigurationStatus,
  ConfigurationActorRef,
  DesiredConfigurationValue,
} from "@rayvan/core";

import {
  ConfigurationDomainError,
  ConfigurationKeyNotFoundError,
} from "./errors.js";
import type { AppliedConfigurationStateRepository } from "./applied-repository.js";
import type { DesiredConfigurationValueRepository } from "./desired-repository.js";
import type { ConfigurationKeyRepository } from "./key-repository.js";
import { assertNoPlaintextSecretDesiredForKey } from "./validation.js";

export interface SaveDesiredConfigurationInput {
  configurationKeyId: string;
  environmentId: string;
  projectId: string;
  desiredValue?: string;
  secretValueRef?: string;
  valueFingerprint?: string;
  updatedBy: ConfigurationActorRef;
  /** Required when updating an existing desired value. */
  expectedRevision?: number;
}

export interface RecordAppliedConfigurationInput {
  configurationKeyId: string;
  environmentId: string;
  projectId: string;
  resourceBindingId: string;
  desiredRevision: number;
  appliedFingerprint?: string;
  applyExecutionId: string;
  verificationExecutionId?: string;
  status: AppliedConfigurationStatus;
  appliedAt?: string;
  verifiedAt?: string;
}

/**
 * Persists desired (key × environment) and applied (key × environment × binding)
 * configuration state. Key/occurrence CRUD stays on ConfigurationService.
 */
export class ConfigurationDesiredStateService {
  constructor(
    private readonly keys: ConfigurationKeyRepository,
    private readonly desired: DesiredConfigurationValueRepository,
    private readonly applied: AppliedConfigurationStateRepository,
  ) {}

  async saveDesired(
    input: SaveDesiredConfigurationInput,
  ): Promise<DesiredConfigurationValue> {
    const key = await this.keys.getById(input.configurationKeyId);
    if (!key) {
      throw new ConfigurationKeyNotFoundError(input.configurationKeyId);
    }

    assertNoPlaintextSecretDesiredForKey(key, {
      desiredValue: input.desiredValue,
    });

    const existing = await this.desired.getByKeyAndEnvironment(
      input.configurationKeyId,
      input.environmentId,
    );

    if (!existing) {
      return this.desired.create({
        configurationKeyId: input.configurationKeyId,
        environmentId: input.environmentId,
        projectId: input.projectId,
        desiredValue: input.desiredValue,
        secretValueRef: input.secretValueRef,
        valueFingerprint: input.valueFingerprint,
        updatedBy: input.updatedBy,
      });
    }

    if (input.expectedRevision === undefined) {
      throw new ConfigurationDomainError(
        "validation_failed",
        "expectedRevision is required when updating desired configuration value",
      );
    }

    return this.desired.updateWithExpectedRevision(existing.id, {
      desiredValue: input.desiredValue ?? null,
      secretValueRef: input.secretValueRef ?? null,
      valueFingerprint: input.valueFingerprint ?? null,
      updatedBy: input.updatedBy,
      expectedRevision: input.expectedRevision,
    });
  }

  getDesired(
    configurationKeyId: string,
    environmentId: string,
  ): Promise<DesiredConfigurationValue | null> {
    return this.desired.getByKeyAndEnvironment(
      configurationKeyId,
      environmentId,
    );
  }

  listByEnvironment(
    environmentId: string,
  ): Promise<DesiredConfigurationValue[]> {
    return this.desired.listByEnvironmentId(environmentId);
  }

  listByProject(projectId: string): Promise<DesiredConfigurationValue[]> {
    return this.desired.listByProjectId(projectId);
  }

  recordApplied(
    input: RecordAppliedConfigurationInput,
  ): Promise<AppliedConfigurationState> {
    return this.applied.upsert({
      configurationKeyId: input.configurationKeyId,
      environmentId: input.environmentId,
      projectId: input.projectId,
      resourceBindingId: input.resourceBindingId,
      desiredRevision: input.desiredRevision,
      appliedFingerprint: input.appliedFingerprint,
      applyExecutionId: input.applyExecutionId,
      verificationExecutionId: input.verificationExecutionId,
      status: input.status,
      appliedAt: input.appliedAt ?? new Date().toISOString(),
      verifiedAt: input.verifiedAt,
    });
  }

  listAppliedByEnvironment(
    environmentId: string,
  ): Promise<AppliedConfigurationState[]> {
    return this.applied.listByEnvironmentId(environmentId);
  }

  listAppliedByProject(
    projectId: string,
  ): Promise<AppliedConfigurationState[]> {
    return this.applied.listByProjectId(projectId);
  }
}
