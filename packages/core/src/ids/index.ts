export type WorkspaceId = string & { readonly __brand: "WorkspaceId" };
export type ProjectId = string & { readonly __brand: "ProjectId" };
export type EnvironmentId = string & { readonly __brand: "EnvironmentId" };
export type IntegrationId = string & { readonly __brand: "IntegrationId" };
export type ConfigurationEntryId = string & {
  readonly __brand: "ConfigurationEntryId";
};
export type ConfigurationKeyId = string & {
  readonly __brand: "ConfigurationKeyId";
};
export type ConfigurationOccurrenceId = string & {
  readonly __brand: "ConfigurationOccurrenceId";
};
export type DesiredConfigurationValueId = string & {
  readonly __brand: "DesiredConfigurationValueId";
};
export type AppliedConfigurationStateId = string & {
  readonly __brand: "AppliedConfigurationStateId";
};
export type FindingId = string & { readonly __brand: "FindingId" };
export type ActionPlanId = string & { readonly __brand: "ActionPlanId" };

export function workspaceId(value: string): WorkspaceId {
  return value as WorkspaceId;
}

export function projectId(value: string): ProjectId {
  return value as ProjectId;
}

export function environmentId(value: string): EnvironmentId {
  return value as EnvironmentId;
}

export function integrationId(value: string): IntegrationId {
  return value as IntegrationId;
}

export function configurationEntryId(value: string): ConfigurationEntryId {
  return value as ConfigurationEntryId;
}

export function configurationKeyId(value: string): ConfigurationKeyId {
  return value as ConfigurationKeyId;
}

export function configurationOccurrenceId(
  value: string,
): ConfigurationOccurrenceId {
  return value as ConfigurationOccurrenceId;
}

export function desiredConfigurationValueId(
  value: string,
): DesiredConfigurationValueId {
  return value as DesiredConfigurationValueId;
}

export function appliedConfigurationStateId(
  value: string,
): AppliedConfigurationStateId {
  return value as AppliedConfigurationStateId;
}

export function findingId(value: string): FindingId {
  return value as FindingId;
}

export function actionPlanId(value: string): ActionPlanId {
  return value as ActionPlanId;
}
