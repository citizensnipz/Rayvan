export type RunState = "idle" | "initializing" | "running" | "validation" | "diagnostics" | "completed" | "failed" | "cancelled" | "interrupted";

export interface ExperimentConfig {
  schema_version: number;
  name: string;
  notes: string;
  tags: string[];
  projection_targets: number[];
  suite: string;
  architecture: string;
  experts: Record<string, number>;
  routing: Record<string, number | string | boolean | null>;
  model: Record<string, number | string | boolean>;
  training: Record<string, number | string>;
}

export interface SchemaOption { id: string; label: string; description?: string; tasks?: string[] }
export interface ResearchSchema {
  schema_version: number;
  suites: SchemaOption[];
  architectures: SchemaOption[];
  expert_families: SchemaOption[];
  presets: Record<string, { label: string; tokens: number }>;
  model_presets: Record<string, Record<string, number>>;
  defaults: ExperimentConfig;
}

export interface ResearchEvent {
  schema_version: number;
  type: string;
  timestamp: string;
  run_id: string;
  [key: string]: unknown;
}

export interface RunSummary {
  run_id: string;
  name: string;
  suite: string;
  architecture: string;
  status: RunState;
  started_at?: string;
  completed_at?: string;
  experts?: Record<string, number>;
  tags?: string[];
  headline?: Record<string, unknown>;
  training_result?: Record<string, unknown>;
  geometric_routing?: Record<string, unknown> | null;
  git?: { commit?: string; dirty?: boolean };
  runDirectory?: string;
}

export interface RunDetail {
  runId: string;
  runDirectory: string;
  config?: ExperimentConfig;
  metadata?: Record<string, unknown>;
  model?: Record<string, unknown>;
  summary?: RunSummary;
  projections?: { schema_version?: number; perplexity_derivation?: string; predictions?: Array<Record<string, unknown>>; calibration?: Array<Record<string, unknown>> };
  diagnostics?: Record<string, unknown>;
  events: ResearchEvent[];
  logs: string;
}

export interface Estimate {
  total_parameters: number;
  approximate_active_parameters: number;
  approximate_flops_per_token: number;
  expert_count: number;
  module_computations_per_forward: number;
}
