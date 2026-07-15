import type {
  AppliedConfigurationState,
  AppliedConfigurationStatus,
} from "@rayvan/core";

export interface UpsertAppliedConfigurationStateInput {
  configurationKeyId: string;
  environmentId: string;
  projectId: string;
  resourceBindingId: string;
  desiredRevision: number;
  appliedFingerprint?: string;
  applyExecutionId: string;
  verificationExecutionId?: string;
  status: AppliedConfigurationStatus;
  appliedAt: string;
  verifiedAt?: string;
}

export interface AppliedConfigurationStateRepository {
  getById(id: string): Promise<AppliedConfigurationState | null>;
  getByKeyEnvironmentBinding(
    configurationKeyId: string,
    environmentId: string,
    resourceBindingId: string,
  ): Promise<AppliedConfigurationState | null>;
  listByEnvironmentId(
    environmentId: string,
  ): Promise<AppliedConfigurationState[]>;
  listByProjectId(projectId: string): Promise<AppliedConfigurationState[]>;
  upsert(
    input: UpsertAppliedConfigurationStateInput,
  ): Promise<AppliedConfigurationState>;
}
