import type { ProjectId, WorkspaceId } from "../ids/index.js";

export interface Workspace {
  id: WorkspaceId;
  name: string;
  projects: Project[];
}

export interface Project {
  id: ProjectId;
  workspaceId: WorkspaceId;
  name: string;
  rootPath?: string;
}
