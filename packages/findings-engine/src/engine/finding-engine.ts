import type {
  FindingActor,
  FindingDetection,
  FindingEvaluationError,
  FindingEvaluationRunRecord,
  FindingEvaluationScope,
  FindingRecord,
  FindingRuleConfiguration,
  FindingRuleDefinition,
  FindingSeverity,
  ProjectId,
} from "@rayvan/core";
import {
  ACTIVE_FINDING_STATUSES,
  findingEvaluationRunId,
} from "@rayvan/core";

import { createCoreEvaluators } from "../evaluators/index.js";
import type { FindingEvaluator } from "../evaluators/types.js";
import {
  getCoreFindingRule,
  listCoreFindingRules,
} from "../rules/registry.js";
import type {
  FindingEngineOptions,
  FindingEngineRepositories,
  FindingEvaluationResult,
  ProjectFindingsContext,
} from "../types.js";
import { validateDetection } from "../validation.js";
import { matchAndApplyDetections } from "./dedupe.js";

const SYSTEM_ACTOR: FindingActor = { kind: "system", id: "findings-engine" };

function isRuleEnabled(
  ruleId: string,
  configurations: FindingRuleConfiguration[] | undefined,
  defaultEnabled: boolean,
): boolean {
  const config = configurations?.find((item) => item.ruleId === ruleId);
  if (!config) {
    return defaultEnabled;
  }
  return config.enabled;
}

function severityOverrides(
  configurations: FindingRuleConfiguration[] | undefined,
): Map<string, FindingSeverity> {
  const map = new Map<string, FindingSeverity>();
  for (const config of configurations ?? []) {
    if (config.severityOverride) {
      map.set(config.ruleId, config.severityOverride);
    }
  }
  return map;
}

function recordInEvaluationScope(
  record: FindingRecord,
  scope: FindingEvaluationScope,
): boolean {
  if (scope.type === "project") {
    return String(record.projectId) === scope.projectId;
  }
  if (scope.type === "environment") {
    return (
      String(record.projectId) === scope.projectId &&
      record.environmentId === scope.environmentId
    );
  }
  return (
    String(record.projectId) === scope.projectId &&
    record.connectionId === scope.connectionId
  );
}

export interface FindingEngineDeps {
  repositories: FindingEngineRepositories;
  /** Override core evaluators (tests). */
  coreEvaluators?: FindingEvaluator[];
  /** Extra rule definitions (e.g. plugin rules registered by host). */
  extraRules?: FindingRuleDefinition[];
}

export class FindingEngine {
  private readonly repositories: FindingEngineRepositories;
  private readonly coreEvaluators: FindingEvaluator[];
  private readonly rulesById: Map<string, FindingRuleDefinition>;
  /**
   * Per-project evaluation queue. Overlapping scopes for the same project
   * must not race on UNIQUE(project_id, fingerprint).
   */
  private readonly projectLocks = new Map<string, Promise<void>>();

  constructor(deps: FindingEngineDeps) {
    this.repositories = deps.repositories;
    this.coreEvaluators = deps.coreEvaluators ?? createCoreEvaluators();
    this.rulesById = new Map(
      [...listCoreFindingRules(), ...(deps.extraRules ?? [])].map((rule) => [
        rule.id,
        rule,
      ]),
    );
  }

  evaluateProject(
    projectId: string,
    options: FindingEngineOptions,
  ): Promise<FindingEvaluationResult> {
    return this.evaluateScoped({ type: "project", projectId }, options);
  }

  evaluateEnvironment(
    projectId: string,
    environmentId: string,
    options: FindingEngineOptions,
  ): Promise<FindingEvaluationResult> {
    return this.evaluateScoped(
      { type: "environment", projectId, environmentId },
      options,
    );
  }

  evaluateConnection(
    projectId: string,
    connectionId: string,
    options: FindingEngineOptions,
  ): Promise<FindingEvaluationResult> {
    return this.evaluateScoped(
      { type: "connection", projectId, connectionId },
      options,
    );
  }

  private evaluateScoped(
    scope: FindingEvaluationScope,
    options: FindingEngineOptions,
  ): Promise<FindingEvaluationResult> {
    return this.withProjectLock(scope.projectId, () =>
      this.runEvaluation(scope, options),
    );
  }

  private async withProjectLock<T>(
    projectId: string,
    fn: () => Promise<T>,
  ): Promise<T> {
    const key = `project:${projectId}`;
    const previous = this.projectLocks.get(key) ?? Promise.resolve();
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    this.projectLocks.set(
      key,
      previous.then(
        () => gate,
        () => gate,
      ),
    );
    await previous.catch(() => undefined);
    try {
      return await fn();
    } finally {
      release();
    }
  }

  private async runEvaluation(
    scope: FindingEvaluationScope,
    options: FindingEngineOptions,
  ): Promise<FindingEvaluationResult> {
    const now = options.now ?? new Date().toISOString();
    const actor = options.actor ?? SYSTEM_ACTOR;
    const runId = findingEvaluationRunId(crypto.randomUUID());
    const startedAt = now;
    const projectId = scope.projectId;

    const context = await this.resolveContext(projectId, options);
    const evaluators: FindingEvaluator[] = [
      ...this.coreEvaluators,
      ...(options.pluginEvaluators ?? []),
    ];

    const detections: FindingDetection[] = [];
    const errors: FindingEvaluationError[] = [];
    const succeededRuleIds = new Set<string>();
    let evaluatorsRun = 0;
    let evaluatorsFailed = 0;
    let cancelled = Boolean(options.abortSignal?.aborted);

    for (const evaluator of evaluators) {
      if (options.abortSignal?.aborted) {
        cancelled = true;
        break;
      }

      const enabledRuleIds = evaluator.ruleIds.filter((ruleId) => {
        const rule =
          this.rulesById.get(ruleId) ?? getCoreFindingRule(ruleId);
        if (!rule) {
          return isRuleEnabled(ruleId, options.ruleConfigurations, true);
        }
        return isRuleEnabled(
          ruleId,
          options.ruleConfigurations,
          rule.enabledByDefault,
        );
      });

      if (enabledRuleIds.length === 0) {
        continue;
      }

      evaluatorsRun += 1;
      try {
        const produced = await evaluator.evaluate({
          projectId,
          scope,
          context,
          now,
          abortSignal: options.abortSignal,
        });

        if (options.abortSignal?.aborted) {
          cancelled = true;
          break;
        }

        const enabledSet = new Set(enabledRuleIds);
        const filtered = produced.detections.filter((detection) =>
          enabledSet.has(detection.ruleId),
        );
        detections.push(...filtered);
        for (const ruleId of produced.evaluatedRuleIds) {
          if (enabledSet.has(ruleId)) {
            succeededRuleIds.add(ruleId);
          }
        }
      } catch (error) {
        evaluatorsFailed += 1;
        errors.push({
          evaluatorId: evaluator.id,
          pluginId: evaluator.pluginId,
          code: "evaluator_failed",
          safeMessage:
            error instanceof Error
              ? error.message
              : "Evaluator failed with an unknown error",
        });
      }
    }

    if (options.abortSignal?.aborted) {
      cancelled = true;
    }

    const finishedAt = options.now ?? new Date().toISOString();

    // Cancelled runs must not create/update/resolve findings — only record the run.
    if (cancelled) {
      const run: FindingEvaluationRunRecord = {
        id: runId,
        projectId: projectId as ProjectId,
        scope,
        trigger: options.trigger,
        status: "cancelled",
        startedAt,
        finishedAt,
        evaluatorsRun,
        evaluatorsFailed,
        findingsCreated: 0,
        findingsUpdated: 0,
        findingsReopened: 0,
        findingsResolved: 0,
        safeErrors: errors,
      };
      await this.repositories.evaluationRuns.save(run);
      return {
        run,
        detections: [],
        created: [],
        updated: [],
        reopened: [],
        resolved: [],
        errors,
      };
    }

    const knownRuleIds = new Set(this.rulesById.keys());
    for (const detection of detections) {
      if (!knownRuleIds.has(detection.ruleId)) {
        knownRuleIds.add(detection.ruleId);
        this.rulesById.set(detection.ruleId, {
          id: detection.ruleId,
          name: detection.ruleId,
          description: "Plugin or dynamic finding rule",
          source: evaluatorSourceForDetection(detection, options),
          category: "other",
          defaultSeverity: detection.severity ?? "warning",
          enabledByDefault: true,
          supportedObjectTypes: ["project"],
        });
      }
      validateDetection(detection);
    }

    const existing = await this.repositories.findings.list({
      projectId,
      includeResolved: true,
    });
    const existingInScope = existing.filter((record) =>
      recordInEvaluationScope(record, scope),
    );

    for (const detection of detections) {
      if (!this.rulesById.has(detection.ruleId)) {
        this.rulesById.set(detection.ruleId, {
          id: detection.ruleId,
          name: detection.ruleId,
          description: "Dynamic finding rule",
          source: { type: "rayvan" },
          category: "other",
          defaultSeverity: detection.severity ?? "warning",
          enabledByDefault: true,
          supportedObjectTypes: ["project"],
        });
      }
    }

    const match = matchAndApplyDetections({
      detections,
      existing,
      rulesById: this.rulesById,
      severityOverrides: severityOverrides(options.ruleConfigurations),
      actor,
      now,
      evaluationRunId: runId,
      succeededRuleIds,
      existingInScope: existingInScope.filter((record) =>
        ACTIVE_FINDING_STATUSES.includes(record.status),
      ),
    });

    const toSave = [
      ...match.created,
      ...match.updated,
      ...match.reopened,
      ...match.resolved,
    ];
    if (this.repositories.findings.saveMany) {
      await this.repositories.findings.saveMany(toSave);
    } else {
      for (const record of toSave) {
        await this.repositories.findings.save(record);
      }
    }

    for (const event of match.events) {
      await this.repositories.lifecycleEvents.append(event);
    }

    let status: FindingEvaluationRunRecord["status"] = "succeeded";
    if (
      evaluatorsFailed > 0 &&
      detections.length === 0 &&
      evaluatorsRun === evaluatorsFailed
    ) {
      status = "failed";
    } else if (evaluatorsFailed > 0) {
      status = "partially_succeeded";
    }

    const run: FindingEvaluationRunRecord = {
      id: runId,
      projectId: projectId as ProjectId,
      scope,
      trigger: options.trigger,
      status,
      startedAt,
      finishedAt,
      evaluatorsRun,
      evaluatorsFailed,
      findingsCreated: match.created.length,
      findingsUpdated: match.updated.length,
      findingsReopened: match.reopened.length,
      findingsResolved: match.resolved.length,
      safeErrors: errors,
    };
    await this.repositories.evaluationRuns.save(run);

    return {
      run,
      detections,
      created: match.created,
      updated: match.updated,
      reopened: match.reopened,
      resolved: match.resolved,
      errors,
    };
  }

  private async resolveContext(
    projectId: string,
    options: FindingEngineOptions,
  ): Promise<ProjectFindingsContext> {
    if (options.context) {
      if (options.context.projectId !== projectId) {
        throw new Error(
          `Context projectId ${options.context.projectId} does not match ${projectId}`,
        );
      }
      return options.context;
    }
    if (!options.loadContext) {
      throw new Error(
        "FindingEngine requires options.context or options.loadContext",
      );
    }
    return options.loadContext(projectId);
  }
}

function evaluatorSourceForDetection(
  detection: FindingDetection,
  options: FindingEngineOptions,
): FindingRuleDefinition["source"] {
  const plugin = options.pluginEvaluators?.find((evaluator) =>
    evaluator.ruleIds.includes(detection.ruleId),
  );
  if (plugin?.pluginId) {
    return { type: "plugin", pluginId: plugin.pluginId };
  }
  return { type: "rayvan" };
}
