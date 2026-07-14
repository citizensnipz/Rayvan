export type DomainEventType =
  | "workspace.created"
  | "project.added"
  | "environment.created"
  | "integration.connected"
  | "finding.detected"
  | "action.plan_created"
  | "action.approved"
  | "action.executed";

export interface DomainEvent<TPayload = unknown> {
  id: string;
  type: DomainEventType;
  occurredAt: string;
  payload: TPayload;
}
