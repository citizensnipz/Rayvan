import {
  environmentId,
  projectId,
  type Environment,
  type EnvironmentColorToken,
  type EnvironmentIconToken,
  type EnvironmentKind,
  type EnvironmentPresentation,
  type EnvironmentStatus,
} from "@rayvan/core";

import type { LocalDatabaseConnection } from "../database/connection.js";
import {
  EnvironmentNotFoundError,
  EnvironmentPersistenceError,
  InvalidEnvironmentNameError,
} from "./errors.js";
import type {
  CreateEnvironmentInput,
  EnvironmentRepository,
  UpdateEnvironmentInput,
} from "./repository.js";
import {
  validateEnvironmentName,
  validateEnvironmentSlug,
} from "./validation.js";

interface EnvironmentRow {
  id: string;
  project_id: string;
  name: string;
  slug: string;
  kind: EnvironmentKind;
  description: string | null;
  presentation_json: string | null;
  status: EnvironmentStatus;
  created_at: string;
  updated_at: string;
}

const SELECT_COLUMNS = `id, project_id, name, slug, kind, description, presentation_json, status, created_at, updated_at`;

function parsePresentation(
  json: string | null,
): EnvironmentPresentation | undefined {
  if (!json) {
    return undefined;
  }
  try {
    const parsed = JSON.parse(json) as {
      color?: EnvironmentColorToken;
      icon?: EnvironmentIconToken;
    };
    const presentation: EnvironmentPresentation = {};
    if (parsed.color) {
      presentation.color = parsed.color;
    }
    if (parsed.icon) {
      presentation.icon = parsed.icon;
    }
    return Object.keys(presentation).length > 0 ? presentation : undefined;
  } catch {
    return undefined;
  }
}

function serializePresentation(
  presentation: EnvironmentPresentation | undefined,
): string | null {
  if (!presentation || Object.keys(presentation).length === 0) {
    return null;
  }
  return JSON.stringify(presentation);
}

function mapRow(row: EnvironmentRow): Environment {
  return {
    id: environmentId(row.id),
    projectId: projectId(row.project_id),
    name: row.name,
    slug: row.slug,
    kind: row.kind,
    description: row.description ?? undefined,
    presentation: parsePresentation(row.presentation_json),
    status: row.status,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

export class SqliteEnvironmentRepository implements EnvironmentRepository {
  constructor(private readonly connection: LocalDatabaseConnection) {}

  async listByProjectId(
    projectIdValue: string,
    options?: { includeArchived?: boolean },
  ): Promise<Environment[]> {
    try {
      const includeArchived = options?.includeArchived ?? false;
      const rows = includeArchived
        ? (this.connection.raw
            .prepare(
              `SELECT ${SELECT_COLUMNS} FROM environments
               WHERE project_id = ?
               ORDER BY updated_at DESC`,
            )
            .all(projectIdValue) as EnvironmentRow[])
        : (this.connection.raw
            .prepare(
              `SELECT ${SELECT_COLUMNS} FROM environments
               WHERE project_id = ? AND status != 'archived'
               ORDER BY updated_at DESC`,
            )
            .all(projectIdValue) as EnvironmentRow[]);

      return rows.map(mapRow);
    } catch (error) {
      throw new EnvironmentPersistenceError(
        "Failed to list environments",
        error,
      );
    }
  }

  async getById(id: string): Promise<Environment | null> {
    try {
      const row = this.connection.raw
        .prepare(`SELECT ${SELECT_COLUMNS} FROM environments WHERE id = ?`)
        .get(id) as EnvironmentRow | undefined;
      return row ? mapRow(row) : null;
    } catch (error) {
      throw new EnvironmentPersistenceError(
        "Failed to load environment",
        error,
      );
    }
  }

  async getByProjectAndSlug(
    projectIdValue: string,
    slug: string,
  ): Promise<Environment | null> {
    try {
      const row = this.connection.raw
        .prepare(
          `SELECT ${SELECT_COLUMNS} FROM environments
           WHERE project_id = ? AND slug = ?`,
        )
        .get(projectIdValue, slug) as EnvironmentRow | undefined;
      return row ? mapRow(row) : null;
    } catch (error) {
      throw new EnvironmentPersistenceError(
        "Failed to load environment by slug",
        error,
      );
    }
  }

  async getByProjectAndName(
    projectIdValue: string,
    name: string,
  ): Promise<Environment | null> {
    try {
      const row = this.connection.raw
        .prepare(
          `SELECT ${SELECT_COLUMNS} FROM environments
           WHERE project_id = ? AND name = ?`,
        )
        .get(projectIdValue, name) as EnvironmentRow | undefined;
      return row ? mapRow(row) : null;
    } catch (error) {
      throw new EnvironmentPersistenceError(
        "Failed to load environment by name",
        error,
      );
    }
  }

  async create(input: CreateEnvironmentInput): Promise<Environment> {
    const name = validateEnvironmentName(input.name);
    const slug = validateEnvironmentSlug(input.slug);
    const now = new Date().toISOString();
    const id = crypto.randomUUID();
    const description = input.description?.trim() || null;
    const status = input.status ?? "local_only";

    try {
      this.connection.raw
        .prepare(
          `INSERT INTO environments (
             id, project_id, name, slug, kind, description,
             presentation_json, status, created_at, updated_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        )
        .run(
          id,
          input.projectId,
          name,
          slug,
          input.kind,
          description,
          serializePresentation(input.presentation),
          status,
          now,
          now,
        );

      const environment = await this.getById(id);
      if (!environment) {
        throw new EnvironmentPersistenceError(
          "Failed to read environment after create",
        );
      }
      return environment;
    } catch (error) {
      if (
        error instanceof InvalidEnvironmentNameError ||
        error instanceof EnvironmentPersistenceError
      ) {
        throw error;
      }
      throw new EnvironmentPersistenceError(
        "Failed to create environment",
        error,
      );
    }
  }

  async update(
    id: string,
    input: UpdateEnvironmentInput,
  ): Promise<Environment> {
    const existing = await this.getById(id);
    if (!existing) {
      throw new EnvironmentNotFoundError(id);
    }

    const name =
      input.name !== undefined
        ? validateEnvironmentName(input.name)
        : existing.name;
    const slug =
      input.slug !== undefined
        ? validateEnvironmentSlug(input.slug)
        : existing.slug;
    const description =
      input.description !== undefined
        ? input.description.trim() || undefined
        : existing.description;
    const kind = input.kind ?? existing.kind;
    const presentation =
      input.presentation !== undefined
        ? input.presentation
        : existing.presentation;
    const status = input.status ?? existing.status;
    const updatedAt = new Date().toISOString();

    try {
      this.connection.raw
        .prepare(
          `UPDATE environments
           SET name = ?, slug = ?, kind = ?, description = ?,
               presentation_json = ?, status = ?, updated_at = ?
           WHERE id = ?`,
        )
        .run(
          name,
          slug,
          kind,
          description ?? null,
          serializePresentation(presentation),
          status,
          updatedAt,
          id,
        );

      const environment = await this.getById(id);
      if (!environment) {
        throw new EnvironmentNotFoundError(id);
      }
      return environment;
    } catch (error) {
      if (
        error instanceof InvalidEnvironmentNameError ||
        error instanceof EnvironmentNotFoundError
      ) {
        throw error;
      }
      throw new EnvironmentPersistenceError(
        "Failed to update environment",
        error,
      );
    }
  }

  async setStatus(
    id: string,
    status: EnvironmentStatus,
  ): Promise<Environment> {
    return this.update(id, { status });
  }

  async delete(id: string): Promise<void> {
    const existing = await this.getById(id);
    if (!existing) {
      throw new EnvironmentNotFoundError(id);
    }

    try {
      this.connection.raw
        .prepare("DELETE FROM environments WHERE id = ?")
        .run(id);
    } catch (error) {
      throw new EnvironmentPersistenceError(
        "Failed to delete environment",
        error,
      );
    }
  }
}
