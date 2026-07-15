import type { EnvironmentId, ProjectId } from "../ids/index.js";

/**
 * Provider-independent environment kind.
 * `local` is retained for local-only / workstation scopes; `test` covers QA/test.
 */
export type EnvironmentKind =
  | "local"
  | "development"
  | "preview"
  | "staging"
  | "production"
  | "test"
  | "custom";

/**
 * Operational + lifecycle status for a Rayvan Environment.
 * Sync timestamps live on sync-run metadata, not on this entity.
 */
export type EnvironmentStatus =
  | "local_only"
  | "healthy"
  | "attention_required"
  | "syncing"
  | "error"
  | "archived";

/** Controlled presentation tokens — never arbitrary CSS values. */
export type EnvironmentColorToken =
  | "neutral"
  | "blue"
  | "green"
  | "amber"
  | "rose"
  | "violet"
  | "cyan";

export type EnvironmentIconToken =
  | "environment"
  | "local"
  | "development"
  | "preview"
  | "staging"
  | "production"
  | "test"
  | "custom";

export interface EnvironmentPresentation {
  color?: EnvironmentColorToken;
  icon?: EnvironmentIconToken;
}

/**
 * A Rayvan-owned logical environment within a project.
 * Provider resources attach via bindings; this is not a Vercel/Railway/etc. environment.
 */
export interface Environment {
  id: EnvironmentId;
  projectId: ProjectId;
  name: string;
  /** Unique within the project (not globally). */
  slug: string;
  kind: EnvironmentKind;
  description?: string;
  presentation?: EnvironmentPresentation;
  status: EnvironmentStatus;
  createdAt: string;
  updatedAt: string;
}
