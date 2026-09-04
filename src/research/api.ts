import { invoke } from "@tauri-apps/api/core";
import type { Estimate, ExperimentConfig, ResearchSchema, RunDetail, RunSummary } from "./types";

export const getSchema = () => invoke<ResearchSchema>("get_research_schema");
export const estimateExperiment = (config: ExperimentConfig) => invoke<Estimate>("estimate_experiment", { config });
export const startExperiment = (config: ExperimentConfig) => invoke<{ runId: string; runDirectory: string }>("start_experiment", { request: { config } });
export const cancelExperiment = () => invoke<void>("cancel_experiment");
export const getActiveExperiment = () => invoke<{ runId: string; runDirectory: string; cancellationRequested: boolean } | null>("get_active_experiment");
export const listExperiments = () => invoke<RunSummary[]>("list_experiments");
export const getExperiment = (runId: string) => invoke<RunDetail>("get_experiment", { runId });
