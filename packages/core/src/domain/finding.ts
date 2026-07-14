import type {
  EnvironmentId,
  FindingId,
  IntegrationId,
  ProjectId,
} from "../ids/index.js";

export type FindingSeverity = "info" | "warning" | "error" | "critical";

export type FindingCategory =
  | "missing_configuration"
  | "configuration_drift"
  | "deployment_failure"
  | "integration_error"
  | "security"
  | "health";

export interface Finding {
  id: FindingId;
  projectId: ProjectId;
  environmentId?: EnvironmentId;
  integrationId?: IntegrationId;
  severity: FindingSeverity;
  category: FindingCategory;
  title: string;
  description: string;
}
