import type { ChangePlan, PluginExecutionActor } from "@rayvan/plugin-sdk";

import { PluginDomainError } from "../errors.js";
import {
  CHANGE_PLAN_SCHEMA_VERSION,
  type ChangePlanRecord,
} from "../models.js";
import type { ChangePlanRepository } from "../repositories/types.js";

export class ChangePlanService {
  constructor(private readonly plans: ChangePlanRepository) {}

  async create(input: {
    pluginId: string;
    connectionId: string;
    projectId: string;
    environmentId?: string;
    resourceBindingId: string;
    desiredStateRevision?: number;
    observedStateChecksum?: string;
    plan: ChangePlan;
    createdBy: PluginExecutionActor;
    expiresAt?: string;
    supersedePlanId?: string;
  }): Promise<ChangePlanRecord> {
    const now = new Date().toISOString();
    const record: ChangePlanRecord = {
      id: input.plan.id,
      pluginId: input.pluginId,
      connectionId: input.connectionId,
      projectId: input.projectId,
      environmentId: input.environmentId,
      resourceBindingId: input.resourceBindingId,
      desiredStateRevision: input.desiredStateRevision,
      observedStateChecksum: input.observedStateChecksum,
      planSchemaVersion: CHANGE_PLAN_SCHEMA_VERSION,
      plan: structuredClone(input.plan),
      status: "pending",
      createdBy: input.createdBy,
      createdAt: now,
      expiresAt: input.expiresAt,
    };

    if (input.supersedePlanId) {
      const previous = await this.plans.getById(input.supersedePlanId);
      if (!previous) {
        throw new PluginDomainError(
          `Change plan not found: ${input.supersedePlanId}`,
        );
      }
      const supersedable = new Set([
        "pending",
        "approved",
        "failed",
        "rejected",
      ]);
      if (!supersedable.has(previous.status)) {
        throw new PluginDomainError(
          `Cannot supersede change plan in status ${previous.status}`,
        );
      }
      await this.plans.supersede({
        planId: input.supersedePlanId,
        supersededByPlanId: record.id,
      });
    }

    await this.plans.save(record);
    return record;
  }

  async getById(id: string): Promise<ChangePlanRecord | undefined> {
    return this.plans.getById(id);
  }

  async requireCurrent(planId: string): Promise<ChangePlanRecord> {
    const plan = await this.plans.getById(planId);
    if (!plan) {
      throw new PluginDomainError(`Change plan not found: ${planId}`);
    }
    if (
      plan.status === "superseded" ||
      plan.status === "expired" ||
      plan.status === "rejected"
    ) {
      throw new PluginDomainError(
        `Change plan is not current (status=${plan.status})`,
      );
    }
    if (plan.expiresAt && plan.expiresAt < new Date().toISOString()) {
      await this.plans.setStatus(planId, "expired");
      throw new PluginDomainError("Change plan has expired");
    }
    return plan;
  }
}
