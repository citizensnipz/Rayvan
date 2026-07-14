import type { Project, Workspace } from "@rayvan/core";

import type { ProjectRepository } from "../projects/repository.js";

export type { ProjectRepository } from "../projects/repository.js";
export { SqliteProjectRepository } from "../projects/sqlite-repository.js";

export interface WorkspaceRepository {
  list(): Promise<Workspace[]>;
  save(workspace: Workspace): Promise<void>;
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

  async list(options?: { includeArchived?: boolean }): Promise<Project[]> {
    const includeArchived = options?.includeArchived ?? false;
    return [...this.projects.values()]
      .filter((project) => includeArchived || project.status === "active")
      .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
  }

  async getById(id: string): Promise<Project | null> {
    return this.projects.get(id) ?? null;
  }

  async create(input: {
    name: string;
    description?: string;
  }): Promise<Project> {
    const { validateProjectName } = await import("../projects/validation.js");
    const now = new Date().toISOString();
    const project: Project = {
      id: crypto.randomUUID() as Project["id"],
      name: validateProjectName(input.name),
      description: input.description?.trim() || undefined,
      status: "active",
      createdAt: now,
      updatedAt: now,
    };
    this.projects.set(project.id, project);
    return project;
  }

  async update(
    id: string,
    input: { name?: string; description?: string },
  ): Promise<Project> {
    const { ProjectNotFoundError } = await import("../projects/errors.js");
    const { validateProjectName } = await import("../projects/validation.js");
    const existing = this.projects.get(id);
    if (!existing) {
      throw new ProjectNotFoundError(id);
    }

    const updated: Project = {
      ...existing,
      name:
        input.name !== undefined
          ? validateProjectName(input.name)
          : existing.name,
      description:
        input.description !== undefined
          ? input.description.trim() || undefined
          : existing.description,
      updatedAt: new Date().toISOString(),
    };
    this.projects.set(id, updated);
    return updated;
  }

  async setArchived(id: string, archived: boolean): Promise<Project> {
    const { ProjectNotFoundError } = await import("../projects/errors.js");
    const existing = this.projects.get(id);
    if (!existing) {
      throw new ProjectNotFoundError(id);
    }

    const updated: Project = {
      ...existing,
      status: archived ? "archived" : "active",
      updatedAt: new Date().toISOString(),
    };
    this.projects.set(id, updated);
    return updated;
  }

  async delete(id: string): Promise<void> {
    const { ProjectNotFoundError } = await import("../projects/errors.js");
    if (!this.projects.has(id)) {
      throw new ProjectNotFoundError(id);
    }
    this.projects.delete(id);
  }
}
