import type { Project, Workspace } from "@rayvan/core";

export interface WorkspaceRepository {
  list(): Promise<Workspace[]>;
  save(workspace: Workspace): Promise<void>;
}

export interface ProjectRepository {
  listByWorkspace(workspaceId: string): Promise<Project[]>;
  save(project: Project): Promise<void>;
}

export class InMemoryWorkspaceRepository implements WorkspaceRepository {
  private readonly workspaces = new Map<string, Workspace>();

  async list(): Promise<Workspace[]> {
    return [...this.workspaces.values()];
  }

  async save(workspace: Workspace): Promise<void> {
    this.workspaces.set(workspace.id, workspace);
  }
}

export class InMemoryProjectRepository implements ProjectRepository {
  private readonly projects = new Map<string, Project>();

  async listByWorkspace(workspaceId: string): Promise<Project[]> {
    return [...this.projects.values()].filter(
      (project) => project.workspaceId === workspaceId,
    );
  }

  async save(project: Project): Promise<void> {
    this.projects.set(project.id, project);
  }
}
