import type {
  ChangeApplyRecord,
  ChangeApplyStatus,
  ChangePlanApprovalRecord,
  ChangePlanRecord,
  ChangePlanRejectionRecord,
  ChangeVerificationRecord,
  ChangeVerificationStatus,
} from "../models.js";
import { PluginDomainError } from "../errors.js";
import type {
  ChangeApplyRepository,
  ChangePlanApprovalRepository,
  ChangePlanRepository,
  ChangeVerificationRepository,
} from "../repositories/types.js";
import type {
  ApplyResult,
  ApprovedChangePlan,
  PluginExecutionActor,
  SerializedPluginExecutionError,
  VerificationResult,
} from "@rayvan/plugin-sdk";

export class ChangeApprovalService {
  constructor(
    private readonly plans: ChangePlanRepository,
    private readonly approvals: ChangePlanApprovalRepository,
    private readonly applies: ChangeApplyRepository,
    private readonly verifications: ChangeVerificationRepository,
  ) {}

  async approve(input: {
    changePlanId: string;
    approvedOperationIds: string[];
    destructiveApproval: boolean;
    approvedBy: PluginExecutionActor;
    comment?: string;
  }): Promise<ChangePlanApprovalRecord> {
    const plan = await this.plans.getById(input.changePlanId);
    if (!plan) {
      throw new PluginDomainError(`Change plan not found: ${input.changePlanId}`);
    }
    if (plan.status !== "pending") {
      throw new PluginDomainError(
        `Only pending plans can be approved (status=${plan.status})`,
      );
    }
    if (plan.plan.destructive && !input.destructiveApproval) {
      throw new PluginDomainError(
        "Destructive plans require explicit destructiveApproval",
      );
    }

    const approval: ChangePlanApprovalRecord = {
      id: crypto.randomUUID(),
      changePlanId: input.changePlanId,
      approvedOperationIds: [...input.approvedOperationIds],
      destructiveApproval: input.destructiveApproval,
      approvedBy: input.approvedBy,
      approvedAt: new Date().toISOString(),
      comment: input.comment,
    };

    await this.approvals.approveAndTransitionPlan({
      approval,
      planId: input.changePlanId,
    });
    return approval;
  }

  async reject(input: {
    changePlanId: string;
    rejectedBy: PluginExecutionActor;
    reason?: string;
  }): Promise<ChangePlanRejectionRecord> {
    const plan = await this.plans.getById(input.changePlanId);
    if (!plan) {
      throw new PluginDomainError(`Change plan not found: ${input.changePlanId}`);
    }
    if (plan.status !== "pending") {
      throw new PluginDomainError(
        `Only pending plans can be rejected (status=${plan.status})`,
      );
    }

    const rejection: ChangePlanRejectionRecord = {
      id: crypto.randomUUID(),
      changePlanId: input.changePlanId,
      rejectedBy: input.rejectedBy,
      rejectedAt: new Date().toISOString(),
      reason: input.reason,
    };

    await this.approvals.rejectAndTransitionPlan({
      rejection,
      planId: input.changePlanId,
    });
    return rejection;
  }

  async buildApprovedChangePlan(
    changePlanId: string,
  ): Promise<ApprovedChangePlan> {
    const plan = await this.plans.getById(changePlanId);
    if (!plan) {
      throw new PluginDomainError(`Change plan not found: ${changePlanId}`);
    }
    if (plan.status !== "approved" && plan.status !== "applying") {
      throw new PluginDomainError(
        `Plan is not approved (status=${plan.status})`,
      );
    }
    const approval = await this.approvals.getLatestApproval(changePlanId);
    if (!approval) {
      throw new PluginDomainError(
        `No approval record for plan ${changePlanId}`,
      );
    }

    return {
      plan: structuredClone(plan.plan),
      approvalId: approval.id,
      approvedAt: approval.approvedAt,
      approvedOperationIds: [...approval.approvedOperationIds],
      approvedBy: approval.approvedBy,
      destructiveApproval: approval.destructiveApproval,
    };
  }

  async beginApply(changePlanId: string): Promise<ChangePlanRecord["id"]> {
    const plan = await this.plans.getById(changePlanId);
    if (!plan) {
      throw new PluginDomainError(`Change plan not found: ${changePlanId}`);
    }
    if (plan.status !== "approved") {
      throw new PluginDomainError(
        "Apply cannot start without an active approval",
      );
    }
    const approval = await this.approvals.getLatestApproval(changePlanId);
    if (!approval) {
      throw new PluginDomainError("Apply cannot start without an active approval");
    }
    await this.applies.beginApply(plan.id);
    return plan.id;
  }

  async completeApply(input: {
    changePlanId: string;
    executionId: string;
    status: ChangeApplyStatus;
    result?: ApplyResult;
    error?: SerializedPluginExecutionError;
    startedAt: string;
  }): Promise<ChangeApplyRecord> {
    const plan = await this.plans.getById(input.changePlanId);
    if (!plan) {
      throw new PluginDomainError(`Change plan not found: ${input.changePlanId}`);
    }
    if (plan.status !== "applying" && plan.status !== "approved") {
      throw new PluginDomainError(
        `Plan is not ready to complete apply (status=${plan.status})`,
      );
    }
    const finishedAt = new Date().toISOString();
    const apply: ChangeApplyRecord = {
      id: crypto.randomUUID(),
      changePlanId: plan.id,
      executionId: input.executionId,
      pluginId: plan.pluginId,
      connectionId: plan.connectionId,
      resourceBindingId: plan.resourceBindingId,
      status: input.status,
      result: input.result,
      error: input.error,
      startedAt: input.startedAt,
      finishedAt,
    };
    await this.applies.completeApply({
      planId: plan.id,
      apply,
      planStatus: input.status === "succeeded" ? "applied" : "failed",
    });
    return apply;
  }

  async recordVerification(input: {
    changeApplyId: string;
    executionId: string;
    status: ChangeVerificationStatus;
    result?: VerificationResult;
    error?: SerializedPluginExecutionError;
  }): Promise<ChangeVerificationRecord> {
    const apply = await this.applies.getById(input.changeApplyId);
    if (!apply) {
      throw new PluginDomainError(
        `Apply record not found: ${input.changeApplyId}`,
      );
    }
    const record: ChangeVerificationRecord = {
      id: crypto.randomUUID(),
      changeApplyId: input.changeApplyId,
      executionId: input.executionId,
      status: input.status,
      result: input.result,
      error: input.error,
      verifiedAt: new Date().toISOString(),
    };
    await this.verifications.save(record);
    return record;
  }
}
