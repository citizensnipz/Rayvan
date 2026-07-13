import type {
  ConfigurationSnapshot,
  Resource,
} from "@rayvan/core";

export interface ConnectionTestResult {
  ok: boolean;
  message: string;
}

export interface DiscoveredResource extends Resource {
  readonly _discovered?: true;
}

export interface HealthCheck {
  name: string;
  status: "healthy" | "degraded" | "unhealthy";
  message?: string;
}

export interface HealthSnapshot {
  integrationId: string;
  checks: HealthCheck[];
  collectedAt: string;
}

export interface ActionRequest {
  kind: string;
  parameters: Record<string, unknown>;
}

export interface PluginActionOperation {
  id: string;
  kind: string;
  description: string;
}

export interface PluginActionPlan {
  summary: string;
  operations: PluginActionOperation[];
}

export interface ApprovedPluginActionPlan extends PluginActionPlan {
  approvalId: string;
  actionPlanId: string;
}

export interface ActionExecutionResult {
  ok: boolean;
  message: string;
  auditEvents: Array<{
    type: string;
    occurredAt: string;
    details?: Record<string, unknown>;
  }>;
}

export type { ConfigurationSnapshot };
