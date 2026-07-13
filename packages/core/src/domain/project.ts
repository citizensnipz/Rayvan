import type { ProjectId } from "../ids/index.js";

export type ProjectStatus = "active" | "archived";

export interface Project {
  id: ProjectId;
  name: string;
  description?: string;
  status: ProjectStatus;
  createdAt: string;
  updatedAt: string;
}
