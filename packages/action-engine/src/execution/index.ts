import type { ApprovedActionPlan } from "@rayvan/core";
import { canExecute } from "../approval/index.js";

export interface ExecutionRecord {
  actionPlanId: string;
  startedAt: string;
  completedAt?: string;
  status: "executing" | "completed" | "failed";
  message?: string;
}

export class ActionExecutionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ActionExecutionError";
  }
}

export function assertExecutablePlan(
  plan: ApprovedActionPlan,
): asserts plan is ApprovedActionPlan {
  if (!canExecute(plan.status, plan.approval)) {
    throw new ActionExecutionError(
      "Action plan cannot execute without explicit approval",
    );
  }
}
