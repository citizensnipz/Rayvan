export interface ApprovalPreparationRequest {
  actionPlanId: string;
  summary: string;
  preparedBy: string;
}

export function prepareApprovalRequest(
  input: ApprovalPreparationRequest,
): ApprovalPreparationRequest {
  return input;
}
