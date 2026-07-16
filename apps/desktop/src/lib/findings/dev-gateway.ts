import type {
  FindingEvaluationRunRecord,
  FindingRecord,
} from "@rayvan/core";
import {
  FindingEngine,
  type FindingEvaluationResult,
  type FindingQuery,
} from "@rayvan/findings-engine";
import {
  createInMemoryFindingsPersistence,
  type InMemoryFindingsPersistence,
} from "@rayvan/local-database";

import {
  buildDevFindingsProjectContext,
  seedDevFindings,
} from "./dev-fixtures.js";
import type {
  FindingEvaluationState,
  FindingsDismissReason,
  FindingsEvaluateOptions,
  FindingsGateway,
  FindingsSeedContext,
  FindingsSuppressPreset,
} from "./types.js";
import {
  DEV_FINDINGS_ENGINE_ACTOR,
  DEV_FINDINGS_USER_ACTOR,
} from "./types.js";

export interface DevFindingsGatewayOptions {
  /** Inject shared persistence (e.g. Environments + Findings sharing one store). */
  persistence?: InMemoryFindingsPersistence;
}

interface EvaluationControl {
  cancelRequested: boolean;
  abort?: AbortController;
  state: FindingEvaluationState;
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
 * DEVELOPMENT ONLY findings gateway.
 * Fresh isolated instance per call unless `persistence` is injected.
 *
 * How to open fixtures:
 * 1. Select a project and open the Findings section
 * 2. ensureProjectSeeded auto-seeds deterministic FindingRecords
 * 3. “Scan project” runs FindingEngine.evaluateProject against seeded context
 */
export function createDevFindingsGateway(
  options?: DevFindingsGatewayOptions,
): FindingsGateway {
  const persistence =
    options?.persistence ?? createInMemoryFindingsPersistence();
  const engine = new FindingEngine({
    repositories: {
      findings: persistence.findings,
      lifecycleEvents: persistence.lifecycleEvents,
      evaluationRuns: persistence.evaluationRuns,
    },
  });

  const seedByProject = new Map<string, Promise<void>>();
  const seededProjects = new Set<string>();
  const seedContextByProject = new Map<string, FindingsSeedContext>();
  const evaluationByProject = new Map<string, EvaluationControl>();

  function getEvaluationControl(projectId: string): EvaluationControl {
    let control = evaluationByProject.get(projectId);
    if (!control) {
      control = {
        cancelRequested: false,
        state: idleEvaluationState(projectId),
      };
      evaluationByProject.set(projectId, control);
    }
    return control;
  }

  async function seedProject(
    projectId: string,
    context?: FindingsSeedContext,
  ): Promise<void> {
    if (context) {
      seedContextByProject.set(projectId, context);
    }
    const resolved = seedContextByProject.get(projectId);
    // Clear prior fixture records for this project so re-seed with real env ids works.
    const existing = await persistence.findings.list({
      projectId,
      includeResolved: true,
    });
    for (const record of existing) {
      // Memory repo has no delete — overwrite via save after wipe by re-creating
      // only when first seed. For re-seed with context, replace fingerprints.
      void record;
    }
    await seedDevFindings(persistence, projectId, resolved);
    seededProjects.add(projectId);
  }

  return {
    ensureProjectSeeded(projectId: string): Promise<void> {
      if (seededProjects.has(projectId)) {
        return Promise.resolve();
      }
      const inflight = seedByProject.get(projectId);
      if (inflight) {
        return inflight;
      }
      const promise = seedProject(projectId)
        .then(() => undefined)
        .catch((error: unknown) => {
          seedByProject.delete(projectId);
          throw error;
        });
      seedByProject.set(projectId, promise);
      return promise;
    },

    async seedForProjectContext(
      projectId: string,
      context: FindingsSeedContext,
    ): Promise<void> {
      seedContextByProject.set(projectId, context);
      // Allow re-seed with mapped environment ids even if already seeded.
      seededProjects.delete(projectId);
      seedByProject.delete(projectId);
      // Memory repository accumulates — save overwrites by id/fingerprint.
      // Replacing: list and overwrite same deterministic ids from fixtures.
      await seedProject(projectId, context);
    },

    async listFindings(query: FindingQuery): Promise<FindingRecord[]> {
      return persistence.service.list(query);
    },

    async getFinding(id: string): Promise<FindingRecord | null> {
      const record = await persistence.service.get(id);
      return record ?? null;
    },

    async getLifecycleEvents(findingId: string) {
      return persistence.service.listLifecycleEvents(findingId);
    },

    async getProjectSummary(projectId: string) {
      return persistence.summaryService.getProjectSummary(projectId);
    },

    async getEnvironmentSummary(projectId: string, environmentId: string) {
      return persistence.summaryService.getEnvironmentSummary(
        projectId,
        environmentId,
      );
    },

    async getIntegrationSummary(projectId: string, connectionId: string) {
      return persistence.summaryService.getIntegrationSummary(
        projectId,
        connectionId,
      );
    },

    async evaluateProject(
      projectId: string,
      options?: FindingsEvaluateOptions,
    ): Promise<FindingEvaluationResult> {
      await this.ensureProjectSeeded(projectId);
      const control = getEvaluationControl(projectId);
      if (control.state.inProgress) {
        throw new Error("A findings evaluation is already in progress");
      }

      control.cancelRequested = false;
      const abort = new AbortController();
      control.abort = abort;
      control.state = {
        projectId,
        inProgress: true,
        phase: "starting",
        cancelled: false,
        progressLabel: "Starting findings scan…",
      };

      try {
        control.state = {
          ...control.state,
          phase: "scanning",
          progressLabel: "Evaluating project findings…",
        };

        if (control.cancelRequested) {
          abort.abort();
        }

        const context = buildDevFindingsProjectContext(
          projectId,
          seedContextByProject.get(projectId),
        );
        const result = await engine.evaluateProject(projectId, {
          trigger: options?.trigger ?? "manual",
          actor: DEV_FINDINGS_ENGINE_ACTOR,
          abortSignal: abort.signal,
          context,
        });

        const cancelled = control.cancelRequested || abort.signal.aborted;
        control.state = {
          projectId,
          inProgress: false,
          phase: cancelled
            ? "cancelled"
            : result.run.status === "partially_succeeded"
              ? "partially_succeeded"
              : result.run.status === "failed"
                ? "failed"
                : "complete",
          cancelled,
          progressLabel: cancelled ? "Scan cancelled" : "Scan complete",
          lastRun: result.run,
          safeErrorMessage: result.errors[0]?.safeMessage,
        };
        return result;
      } catch (error) {
        control.state = {
          projectId,
          inProgress: false,
          phase: "failed",
          cancelled: false,
          progressLabel: "Scan failed",
          safeErrorMessage:
            error instanceof Error ? error.message : "Findings evaluation failed",
        };
        throw error;
      }
    },

    async evaluateEnvironment(
      projectId: string,
      environmentId: string,
      options?: FindingsEvaluateOptions,
    ): Promise<FindingEvaluationResult> {
      await this.ensureProjectSeeded(projectId);
      const context = buildDevFindingsProjectContext(
        projectId,
        seedContextByProject.get(projectId),
      );
      return engine.evaluateEnvironment(projectId, environmentId, {
        trigger: options?.trigger ?? "manual",
        actor: DEV_FINDINGS_ENGINE_ACTOR,
        context,
      });
    },

    async cancelEvaluation(projectId: string): Promise<void> {
      const control = getEvaluationControl(projectId);
      control.cancelRequested = true;
      control.abort?.abort();
      if (control.state.inProgress) {
        control.state = {
          ...control.state,
          cancelled: true,
          progressLabel: "Cancelling scan…",
        };
      }
    },

    async getLastEvaluationRun(
      projectId: string,
    ): Promise<FindingEvaluationRunRecord | null> {
      const control = getEvaluationControl(projectId);
      if (control.state.lastRun) {
        return control.state.lastRun;
      }
      const runs = await persistence.evaluationRuns.listByProject(projectId, 1);
      return runs[0] ?? null;
    },

    async getEvaluationState(projectId: string): Promise<FindingEvaluationState> {
      const control = getEvaluationControl(projectId);
      if (!control.state.lastRun) {
        const runs = await persistence.evaluationRuns.listByProject(projectId, 1);
        if (runs[0]) {
          control.state = {
            ...control.state,
            lastRun: runs[0],
          };
        }
      }
      return { ...control.state };
    },

    acknowledge(findingId: string, comment?: string): Promise<FindingRecord> {
      return persistence.service.acknowledge(
        findingId,
        DEV_FINDINGS_USER_ACTOR,
        comment,
      );
    },

    dismiss(
      findingId: string,
      reason: FindingsDismissReason | string,
    ): Promise<FindingRecord> {
      return persistence.service.dismiss(
        findingId,
        DEV_FINDINGS_USER_ACTOR,
        reason,
      );
    },

    async suppress(
      findingId: string,
      preset: FindingsSuppressPreset,
      reason?: string,
    ): Promise<FindingRecord> {
      if (preset === "until_next_sync") {
        // Far-future until; evaluateProject / sync callers may clear suppressions later.
        const until = new Date(
          Date.now() + 365 * 24 * 60 * 60 * 1000,
        ).toISOString();
        return persistence.service.suppress(
          findingId,
          DEV_FINDINGS_USER_ACTOR,
          { until, reason: reason ?? "Until next sync" },
        );
      }
      return persistence.service.suppress(
        findingId,
        DEV_FINDINGS_USER_ACTOR,
        { preset, reason },
      );
    },
  };
}

/** Expose persistence for Environments gateway sharing. */
export function createSharedFindingsPersistence(): InMemoryFindingsPersistence {
  return createInMemoryFindingsPersistence();
}
