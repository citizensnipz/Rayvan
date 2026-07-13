import type { EnvironmentId, ProjectId } from "../ids/index.js";

export type EnvironmentKind =
  | "local"
  | "development"
  | "preview"
  | "staging"
  | "production"
  | "custom";

export interface Environment {
  id: EnvironmentId;
  projectId: ProjectId;
  name: string;
  kind: EnvironmentKind;
}
