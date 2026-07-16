import type {
  FindingEvaluationRunRecord,
  FindingLifecycleEventRecord,
  FindingRecord,
  FindingSummary,
} from "@rayvan/core";
import type {
  FindingEvaluationResult,
  FindingQuery,
} from "@rayvan/findings-engine";

import { desktopDaemon } from "../daemon/client.js";
import type {
  FindingEvaluationState,
  FindingsDismissReason,
  FindingsEvaluateOptions,
  FindingsGateway,
  FindingsSuppressPreset,
} from "./types.js";

function emptySummary(projectId?: string): FindingSummary {
  void projectId;
  return {
    openCount: 0,
    acknowledgedCount: 0,
    bySeverity: {
      info: 0,
      warning: 0,
      error: 0,
      critical: 0,
    },
    byCategory: {},
    hasRemediableFindings: false,
  };
}

function normalizeSummary(value: unknown): FindingSummary {
  if (!value || typeof value !== "object") {
    return emptySummary();
  }
  const record = value as Partial<FindingSummary> & {
    total?: number;
    byStatus?: Record<string, number>;
  };
  if (typeof record.openCount === "number") {
    return {
      openCount: record.openCount,
      acknowledgedCount: record.acknowledgedCount ?? 0,
      bySeverity: record.bySeverity ?? emptySummary().bySeverity,
      byCategory: record.byCategory ?? {},
      highestSeverity: record.highestSeverity,
      hasRemediableFindings: record.hasRemediableFindings ?? false,
    };
  }
  return {
    openCount: record.byStatus?.open ?? record.total ?? 0,
    acknowledgedCount: record.byStatus?.acknowledged ?? 0,
    bySeverity: {
      info: Number(record.bySeverity?.info ?? 0),
      warning: Number(record.bySeverity?.warning ?? 0),
      error: Number(record.bySeverity?.error ?? 0),
      critical: Number(record.bySeverity?.critical ?? 0),
    },
    byCategory: {},
    hasRemediableFindings: false,
  };
}

function idleEvaluationState(projectId: string): FindingEvaluationState {
  return {
    projectId,
    inProgress: false,
    phase: "idle",
    cancelled: false,
  };
}

/**
 * Daemon-backed findings gateway. Prefer when `rayvand` is connected.
 * Lifecycle events and full local engine evaluation shapes are best-effort.
 */
export function createDaemonFindingsGateway(): FindingsGateway {
  const evaluationByProject = new Map<string, FindingEvaluationState>();

  return {
    async ensureProjectSeeded(): Promise<void> {
      // Production path — no fixture seeding.
    },

    async listFindings(query: FindingQuery): Promise<FindingRecord[]> {
      const findings = await desktopDaemon.listFindings({
        projectId: query.projectId,
        environmentId: query.environmentId,
        status: query.statuses?.[0],
      });
      return (findings as FindingRecord[]) ?? [];
    },

    async getFinding(id: string): Promise<FindingRecord | null> {
      return (await desktopDaemon.getFinding(id)) as FindingRecord | null;
    },

    async getLifecycleEvents(): Promise<FindingLifecycleEventRecord[]> {
      return [];
    },

    async getProjectSummary(projectId: string): Promise<FindingSummary> {
      return normalizeSummary(
        await desktopDaemon.getFindingSummary({ projectId }),
      );
    },

    async getEnvironmentSummary(
      projectId: string,
      environmentId: string,
    ): Promise<FindingSummary> {
      return normalizeSummary(
        await desktopDaemon.getFindingSummary({ projectId, environmentId }),
      );
    },

    async getIntegrationSummary(
      projectId: string,
      connectionId: string,
    ): Promise<FindingSummary> {
      return normalizeSummary(
        await desktopDaemon.getFindingSummary({
          projectId,
          integrationId: connectionId,
        }),
      );
    },

    async evaluateProject(
      projectId: string,
      _options?: FindingsEvaluateOptions,
    ): Promise<FindingEvaluationResult> {
      void _options;
      evaluationByProject.set(projectId, {
        projectId,
        inProgress: true,
        phase: "scanning",
        cancelled: false,
        progressLabel: "Scanning via daemon…",
      });
      try {
        const operation = await desktopDaemon.scanFindings({ projectId });
        const findings = (await desktopDaemon.listFindings({
          projectId,
        })) as FindingRecord[];
        const run: FindingEvaluationRunRecord = {
          id: operation.id as FindingEvaluationRunRecord["id"],
          projectId: projectId as FindingEvaluationRunRecord["projectId"],
          scope: { type: "project", projectId: projectId as never },
          trigger: "manual",
          status:
            operation.status === "succeeded"
              ? "succeeded"
              : operation.status === "failed"
                ? "failed"
                : "partially_succeeded",
          startedAt: operation.startedAt ?? new Date().toISOString(),
          finishedAt: operation.finishedAt ?? new Date().toISOString(),
          evaluatorsRun: 0,
          evaluatorsFailed: 0,
          findingsCreated: findings.length,
          findingsUpdated: 0,
          findingsReopened: 0,
          findingsResolved: 0,
          safeErrors: operation.safeError
            ? [
                {
                  evaluatorId: "daemon-scan",
                  safeMessage: operation.safeError.message,
                  code: operation.safeError.code,
                },
              ]
            : [],
        };
        evaluationByProject.set(projectId, {
          projectId,
          inProgress: false,
          phase: run.status === "failed" ? "failed" : "complete",
          cancelled: false,
          lastRun: run,
        });
        return {
          run,
          detections: [],
          created: findings,
          updated: [],
          reopened: [],
          resolved: [],
          errors: run.safeErrors,
        };
      } catch (error) {
        evaluationByProject.set(projectId, {
          projectId,
          inProgress: false,
          phase: "failed",
          cancelled: false,
          safeErrorMessage:
            error instanceof Error ? error.message : "Scan failed",
        });
        throw error;
      }
    },

    async evaluateEnvironment(
      projectId: string,
      environmentId: string,
      options?: FindingsEvaluateOptions,
    ): Promise<FindingEvaluationResult> {
      void environmentId;
      return this.evaluateProject(projectId, options);
    },

    async cancelEvaluation(projectId: string): Promise<void> {
      const current = evaluationByProject.get(projectId);
      if (current) {
        evaluationByProject.set(projectId, {
          ...current,
          inProgress: false,
          phase: "cancelled",
          cancelled: true,
        });
      }
    },

    async getLastEvaluationRun(
      projectId: string,
    ): Promise<FindingEvaluationRunRecord | null> {
      return evaluationByProject.get(projectId)?.lastRun ?? null;
    },

    async getEvaluationState(
      projectId: string,
    ): Promise<FindingEvaluationState> {
      return evaluationByProject.get(projectId) ?? idleEvaluationState(projectId);
    },

    async acknowledge(findingId: string, comment?: string): Promise<FindingRecord> {
      return (await desktopDaemon.acknowledgeFinding(
        findingId,
        comment,
      )) as FindingRecord;
    },

    async dismiss(
      findingId: string,
      reason: FindingsDismissReason | string,
    ): Promise<FindingRecord> {
      return (await desktopDaemon.dismissFinding(
        findingId,
        reason,
      )) as FindingRecord;
    },

    async suppress(
      findingId: string,
      _preset: FindingsSuppressPreset,
      reason?: string,
    ): Promise<FindingRecord> {
      return (await desktopDaemon.suppressFinding(
        findingId,
        reason,
      )) as FindingRecord;
    },

    async seedForProjectContext(): Promise<void> {
      // no-op
    },
  };
}
