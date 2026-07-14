import { invoke } from "@tauri-apps/api/core";
import { projectId, type Project } from "@rayvan/core";
import type { ProjectRepository } from "@rayvan/local-database";
import {
  InvalidProjectNameError,
  ProjectNotFoundError,
  ProjectPersistenceError,
} from "@rayvan/local-database";

interface CommandError {
  code: string;
  message: string;
  id?: string;
}

interface NativeProject {
  id: string;
  name: string;
  description?: string;
  status: "active" | "archived";
  createdAt: string;
  updatedAt: string;
}

function mapProject(project: NativeProject): Project {
  return {
    id: projectId(project.id),
    name: project.name,
    description: project.description,
    status: project.status,
    createdAt: project.createdAt,
    updatedAt: project.updatedAt,
  };
}

function mapCommandError(error: CommandError): Error {
  switch (error.code) {
    case "not_found":
      return new ProjectNotFoundError(error.id ?? error.message);
    case "validation_failed":
      return new InvalidProjectNameError();
    default:
      return new ProjectPersistenceError(error.message);
  }
}

async function invokeCommand<T>(
  command: string,
  payload?: Record<string, unknown>,
): Promise<T> {
  try {
    return await invoke<T>(command, payload);
  } catch (error) {
    if (
      typeof error === "object" &&
      error !== null &&
      "code" in error &&
      "message" in error
    ) {
      throw mapCommandError(error as CommandError);
    }
    throw new ProjectPersistenceError(
      error instanceof Error ? error.message : "Unknown persistence error",
      error,
    );
  }
}

export class TauriProjectRepository implements ProjectRepository {
  async list(options?: { includeArchived?: boolean }): Promise<Project[]> {
    const projects = await invokeCommand<NativeProject[]>("list_projects", {
      includeArchived: options?.includeArchived ?? false,
    });
    return projects.map(mapProject);
  }

  async getById(id: string): Promise<Project | null> {
    const project = await invokeCommand<NativeProject | null>("get_project", {
      id,
    });
    return project ? mapProject(project) : null;
  }

  async create(input: {
    name: string;
    description?: string;
  }): Promise<Project> {
    const project = await invokeCommand<NativeProject>("create_project", input);
    return mapProject(project);
  }

  async update(
    id: string,
    input: { name?: string; description?: string },
  ): Promise<Project> {
    const project = await invokeCommand<NativeProject>("update_project", {
      id,
      ...input,
    });
    return mapProject(project);
  }

  async setArchived(id: string, archived: boolean): Promise<Project> {
    const project = await invokeCommand<NativeProject>("set_project_archived", {
      id,
      archived,
    });
    return mapProject(project);
  }

  async delete(id: string): Promise<void> {
    await invokeCommand<void>("delete_project", { id });
  }
}

export const tauriProjectRepository = new TauriProjectRepository();
