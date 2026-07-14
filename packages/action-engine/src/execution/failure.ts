export interface ActionFailureResult {
  actionPlanId: string;
  failedAt: string;
  message: string;
  retryable: boolean;
}

export function createFailureResult(
  input: Omit<ActionFailureResult, "failedAt"> & { failedAt?: string },
): ActionFailureResult {
  return {
    ...input,
    failedAt: input.failedAt ?? new Date().toISOString(),
  };
}
