import type {
  FindingActor,
  FindingEvaluationRunRecord,
  FindingLifecycleEventRecord,
  FindingRecord,
  FindingSummary,
} from "@rayvan/core";
import type {
  FindingEvaluationResult,
  FindingQuery,
} from "@rayvan/findings-engine";

export type FindingEvaluationPhase =
  | "idle"
  | "starting"
  | "scanning"
  | "complete"
  | "cancelled"
  | "failed"
  | "partially_succeeded";

export interface FindingEvaluationState {
  projectId: string;
  inProgress: boolean;
  phase: FindingEvaluationPhase;
  cancelled: boolean;
  progressLabel?: string;
  lastRun?: FindingEvaluationRunRecord;
  safeErrorMessage?: string;
}

export type FindingsDismissReason =
  | "Expected behaviour"
  | "False positive"
  | "Not relevant"
  | "Will handle outside Rayvan"
  | "Other";

export type FindingsSuppressPreset = "until_next_sync" | "24h" | "7d";

export interface FindingsEvaluateOptions {
  trigger?: FindingEvaluationRunRecord["trigger"];
}

/**
 * Host-side abstraction for the Findings workspace.
 * React-free; implementations must not call real provider APIs or mutate infra.
 */
export interface FindingsGateway {
  ensureProjectSeeded(projectId: string): Promise<void>;

  listFindings(query: FindingQuery): Promise<FindingRecord[]>;
  getFinding(id: string): Promise<FindingRecord | null>;
  getLifecycleEvents(findingId: string): Promise<FindingLifecycleEventRecord[]>;

  getProjectSummary(projectId: string): Promise<FindingSummary>;
  getEnvironmentSummary(
    projectId: string,
    environmentId: string,
  ): Promise<FindingSummary>;
  getIntegrationSummary(
    projectId: string,
    connectionId: string,
  ): Promise<FindingSummary>;

  evaluateProject(
    projectId: string,
    options?: FindingsEvaluateOptions,
  ): Promise<FindingEvaluationResult>;
  evaluateEnvironment(
    projectId: string,
    environmentId: string,
    options?: FindingsEvaluateOptions,
  ): Promise<FindingEvaluationResult>;
  cancelEvaluation?(projectId: string): Promise<void>;
  getLastEvaluationRun(
    projectId: string,
  ): Promise<FindingEvaluationRunRecord | null>;
  getEvaluationState(projectId: string): Promise<FindingEvaluationState>;

  acknowledge(findingId: string, comment?: string): Promise<FindingRecord>;
  dismiss(
    findingId: string,
    reason: FindingsDismissReason | string,
  ): Promise<FindingRecord>;
  suppress(
    findingId: string,
    preset: FindingsSuppressPreset,
    reason?: string,
  ): Promise<FindingRecord>;

  /**
   * Optional: re-seed fixture findings using real environment / connection ids
   * from the Environments gateway after that gateway finishes seeding.
   */
  seedForProjectContext?(
    projectId: string,
    context: FindingsSeedContext,
  ): Promise<void>;
}

export interface FindingsSeedContext {
  environmentIdsByName?: Record<string, string>;
  connectionIdsByPluginId?: Record<string, string>;
}

export const DEV_FINDINGS_USER_ACTOR: FindingActor = {
  kind: "user",
  id: "dev-user",
  displayName: "Developer",
};

export const DEV_FINDINGS_SYSTEM_ACTOR: FindingActor = {
  kind: "system",
  id: "rayvan-findings",
};

export const DEV_FINDINGS_ENGINE_ACTOR: FindingActor = {
  kind: "rayvan",
  id: "findings-engine",
};

export const FINDINGS_DISMISS_REASONS: readonly FindingsDismissReason[] = [
  "Expected behaviour",
  "False positive",
  "Not relevant",
  "Will handle outside Rayvan",
  "Other",
] as const;

export const FINDINGS_SUPPRESS_PRESETS: ReadonlyArray<{
  id: FindingsSuppressPreset;
  label: string;
}> = [
  { id: "until_next_sync", label: "Until next sync" },
  { id: "24h", label: "24 hours" },
  { id: "7d", label: "7 days" },
] as const;
