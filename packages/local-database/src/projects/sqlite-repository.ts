import { projectId, type Project, type ProjectStatus } from "@rayvan/core";

import type { LocalDatabaseConnection } from "../database/connection.js";
import {
  InvalidProjectNameError,
  ProjectNotFoundError,
  ProjectPersistenceError,
} from "./errors.js";
import type { ProjectRepository } from "./repository.js";
import { validateProjectName } from "./validation.js";

interface ProjectRow {
  id: string;
  name: string;
  description: string | null;
  status: ProjectStatus;
  created_at: string;
  updated_at: string;
}

function mapRow(row: ProjectRow): Project {
  return {
    id: projectId(row.id),
    name: row.name,
    description: row.description ?? undefined,
    status: row.status,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

export class SqliteProjectRepository implements ProjectRepository {
  constructor(private readonly connection: LocalDatabaseConnection) {}

  async list(options?: { includeArchived?: boolean }): Promise<Project[]> {
    try {
      const includeArchived = options?.includeArchived ?? false;
      const rows = includeArchived
        ? (this.connection.raw
            .prepare(
              "SELECT id, name, description, status, created_at, updated_at FROM projects ORDER BY updated_at DESC",
            )
            .all() as ProjectRow[])
        : (this.connection.raw
            .prepare(
              "SELECT id, name, description, status, created_at, updated_at FROM projects WHERE status = 'active' ORDER BY updated_at DESC",
            )
            .all() as ProjectRow[]);

      return rows.map(mapRow);
    } catch (error) {
      throw new ProjectPersistenceError("Failed to list projects", error);
    }
  }

  async getById(id: string): Promise<Project | null> {
    try {
      const row = this.connection.raw
        .prepare(
          "SELECT id, name, description, status, created_at, updated_at FROM projects WHERE id = ?",
        )
        .get(id) as ProjectRow | undefined;

      return row ? mapRow(row) : null;
    } catch (error) {
      throw new ProjectPersistenceError("Failed to load project", error);
    }
  }

  async create(input: {
    name: string;
    description?: string;
  }): Promise<Project> {
    const name = validateProjectName(input.name);
    const now = new Date().toISOString();
    const id = crypto.randomUUID();
    const description = input.description?.trim() || null;

    try {
      this.connection.raw
        .prepare(
          `INSERT INTO projects (id, name, description, status, created_at, updated_at)
           VALUES (?, ?, ?, 'active', ?, ?)`,
        )
        .run(id, name, description, now, now);

      const project = await this.getById(id);
      if (!project) {
        throw new ProjectPersistenceError("Failed to read project after create");
      }
      return project;
    } catch (error) {
      if (error instanceof InvalidProjectNameError) {
        throw error;
      }
      throw new ProjectPersistenceError("Failed to create project", error);
    }
  }

  async update(
    id: string,
    input: { name?: string; description?: string },
  ): Promise<Project> {
    const existing = await this.getById(id);
    if (!existing) {
      throw new ProjectNotFoundError(id);
    }

    const name =
      input.name !== undefined ? validateProjectName(input.name) : existing.name;
    const description =
      input.description !== undefined
        ? input.description.trim() || undefined
        : existing.description;
    const updatedAt = new Date().toISOString();

    try {
      this.connection.raw
        .prepare(
          `UPDATE projects
           SET name = ?, description = ?, updated_at = ?
           WHERE id = ?`,
        )
        .run(name, description ?? null, updatedAt, id);

      const project = await this.getById(id);
      if (!project) {
        throw new ProjectNotFoundError(id);
      }
      return project;
    } catch (error) {
      if (
        error instanceof InvalidProjectNameError ||
        error instanceof ProjectNotFoundError
      ) {
        throw error;
      }
      throw new ProjectPersistenceError("Failed to update project", error);
    }
  }

  async setArchived(id: string, archived: boolean): Promise<Project> {
    const existing = await this.getById(id);
    if (!existing) {
      throw new ProjectNotFoundError(id);
    }

    const status: ProjectStatus = archived ? "archived" : "active";
    const updatedAt = new Date().toISOString();

    try {
      this.connection.raw
        .prepare(
          "UPDATE projects SET status = ?, updated_at = ? WHERE id = ?",
        )
        .run(status, updatedAt, id);

      const project = await this.getById(id);
      if (!project) {
        throw new ProjectNotFoundError(id);
      }
      return project;
    } catch (error) {
      if (error instanceof ProjectNotFoundError) {
        throw error;
      }
      throw new ProjectPersistenceError("Failed to update project status", error);
    }
  }

  async delete(id: string): Promise<void> {
    const existing = await this.getById(id);
    if (!existing) {
      throw new ProjectNotFoundError(id);
    }

    try {
      this.connection.raw.prepare("DELETE FROM projects WHERE id = ?").run(id);
    } catch (error) {
      throw new ProjectPersistenceError("Failed to delete project", error);
    }
  }
}
