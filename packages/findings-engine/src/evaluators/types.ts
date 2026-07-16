import type {
  FindingDetection,
  FindingEvaluationScope,
} from "@rayvan/core";

import type { ProjectFindingsContext } from "../types.js";

export interface FindingEvaluatorInput<TContext = ProjectFindingsContext> {
  projectId: string;
  scope: FindingEvaluationScope;
  context: TContext;
  now: string;
  abortSignal?: AbortSignal;
}

/**
 * Result of a single evaluator pass.
 * `evaluatedRuleIds` are the only rule IDs eligible for auto-resolve —
 * rules skipped due to missing input data must be omitted.
 */
export interface FindingEvaluatorResult {
  detections: FindingDetection[];
  evaluatedRuleIds: readonly string[];
}

/**
 * Pure evaluator that emits FindingDetection[] for a scope.
 * Must not assign FindingRecord IDs or mutate persistence.
 */
export interface FindingEvaluator<TContext = ProjectFindingsContext> {
  id: string;
  /** Rule IDs owned by this evaluator — used for resolve-on-success scoping. */
  ruleIds: readonly string[];
  /** When set, marks this as a plugin evaluator for partial-failure isolation. */
  pluginId?: string;
  evaluate(
    input: FindingEvaluatorInput<TContext>,
  ): Promise<FindingEvaluatorResult> | FindingEvaluatorResult;
}

/** Helper for evaluators / adapters that fully evaluated every enabled rule they own. */
export function evaluatorResult(
  detections: FindingDetection[],
  evaluatedRuleIds: readonly string[],
): FindingEvaluatorResult {
  return { detections, evaluatedRuleIds };
}
