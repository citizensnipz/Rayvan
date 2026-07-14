export interface AuditEvent {
  id: string;
  actionPlanId: string;
  type: "planned" | "approved" | "executed" | "failed" | "cancelled";
  occurredAt: string;
  actor: string;
  details?: Record<string, unknown>;
}

export function createAuditEvent(
  input: Omit<AuditEvent, "occurredAt"> & { occurredAt?: string },
): AuditEvent {
  return {
    ...input,
    occurredAt: input.occurredAt ?? new Date().toISOString(),
  };
}
