import type {
  ConfigurationActorRef,
  DesiredConfigurationValue,
} from "@rayvan/core";

export interface CreateDesiredConfigurationValueInput {
  configurationKeyId: string;
  environmentId: string;
  projectId: string;
  desiredValue?: string;
  secretValueRef?: string;
  valueFingerprint?: string;
  updatedBy: ConfigurationActorRef;
}

export interface UpdateDesiredConfigurationValueInput {
  desiredValue?: string | null;
  secretValueRef?: string | null;
  valueFingerprint?: string | null;
  updatedBy: ConfigurationActorRef;
  expectedRevision: number;
}

export interface DesiredConfigurationValueRepository {
  getById(id: string): Promise<DesiredConfigurationValue | null>;
  getByKeyAndEnvironment(
    configurationKeyId: string,
    environmentId: string,
  ): Promise<DesiredConfigurationValue | null>;
  listByEnvironmentId(
    environmentId: string,
  ): Promise<DesiredConfigurationValue[]>;
  listByProjectId(projectId: string): Promise<DesiredConfigurationValue[]>;
  create(
    input: CreateDesiredConfigurationValueInput,
  ): Promise<DesiredConfigurationValue>;
  updateWithExpectedRevision(
    id: string,
    input: UpdateDesiredConfigurationValueInput,
  ): Promise<DesiredConfigurationValue>;
  delete(id: string): Promise<void>;
}
