import type { ApplyContext } from "../contexts/index.js";
import { PluginValidationError } from "../errors/index.js";
import {
  validateApprovedChangePlan,
  validateChangePlan,
} from "../validation/index.js";

/**
 * Apply-only safety checks that run before the runtime invokes the handler.
 * Failures are mapped to approval_invalid by PluginExecutionService.
 */
export function assertApplyGuards(
  pluginId: string,
  context: ApplyContext,
): void {
  const approved = context.approvedPlan;
  const plan = approved.plan;

  if (plan.pluginId !== pluginId) {
    throw new PluginValidationError(
      `Approved plan pluginId "${plan.pluginId}" does not match request pluginId "${pluginId}"`,
      { pluginId },
    );
  }

  if (plan.resourceId !== context.resource.resourceId) {
    throw new PluginValidationError(
      `Approved plan resourceId "${plan.resourceId}" does not match resource binding "${context.resource.resourceId}"`,
      { pluginId },
    );
  }

  if (!Array.isArray(plan.operations) || plan.operations.length < 1) {
    throw new PluginValidationError(
      "Approved plan must include at least one operation",
      { pluginId },
    );
  }

  validateChangePlan(plan);
  validateApprovedChangePlan(approved);

  const operationIds = new Set(plan.operations.map((operation) => operation.id));
  const approvedIds = new Set(approved.approvedOperationIds);

  for (const operation of plan.operations) {
    if (operation.requiresApproval && !approvedIds.has(operation.id)) {
      throw new PluginValidationError(
        `Operation "${operation.id}" requires approval but is not in approvedOperationIds`,
        { pluginId },
      );
    }
  }

  for (const approvedId of approved.approvedOperationIds) {
    if (!operationIds.has(approvedId)) {
      throw new PluginValidationError(
        `approvedOperationIds contains unknown operation id "${approvedId}"`,
        { pluginId },
      );
    }
  }

  const isDestructive =
    plan.destructive ||
    plan.operations.some((operation) => operation.destructive === true);

  if (isDestructive && approved.destructiveApproval !== true) {
    throw new PluginValidationError(
      "Destructive plan requires destructiveApproval === true",
      { pluginId },
    );
  }
}
