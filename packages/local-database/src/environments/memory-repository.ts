import {
  environmentId,
  projectId,
  type Environment,
  type EnvironmentStatus,
} from "@rayvan/core";

import { EnvironmentNotFoundError } from "./errors.js";
import type {
  CreateEnvironmentInput,
  EnvironmentRepository,
  UpdateEnvironmentInput,
} from "./repository.js";
import {
  validateEnvironmentName,
  validateEnvironmentSlug,
} from "./validation.js";

export class InMemoryEnvironmentRepository implements EnvironmentRepository {
  private readonly environments = new Map<string, Environment>();

  async listByProjectId(
    projectIdValue: string,
    options?: { includeArchived?: boolean },
  ): Promise<Environment[]> {
    const includeArchived = options?.includeArchived ?? false;
    return [...this.environments.values()]
      .filter((environment) => environment.projectId === projectIdValue)
      .filter(
        (environment) =>
          includeArchived || environment.status !== "archived",
      )
      .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
  }

  async getById(id: string): Promise<Environment | null> {
    return this.environments.get(id) ?? null;
  }

  async getByProjectAndSlug(
    projectIdValue: string,
    slug: string,
  ): Promise<Environment | null> {
    return (
      [...this.environments.values()].find(
        (environment) =>
          environment.projectId === projectIdValue &&
          environment.slug === slug,
      ) ?? null
    );
  }

  async getByProjectAndName(
    projectIdValue: string,
    name: string,
  ): Promise<Environment | null> {
    return (
      [...this.environments.values()].find(
        (environment) =>
          environment.projectId === projectIdValue &&
          environment.name === name,
      ) ?? null
    );
  }

  async create(input: CreateEnvironmentInput): Promise<Environment> {
    const now = new Date().toISOString();
    const environment: Environment = {
      id: environmentId(crypto.randomUUID()),
      projectId: projectId(input.projectId),
      name: validateEnvironmentName(input.name),
      slug: validateEnvironmentSlug(input.slug),
      kind: input.kind,
      description: input.description?.trim() || undefined,
      presentation: input.presentation,
      status: input.status ?? "local_only",
      createdAt: now,
      updatedAt: now,
    };
    this.environments.set(environment.id, environment);
    return environment;
  }

  async update(
    id: string,
    input: UpdateEnvironmentInput,
  ): Promise<Environment> {
    const existing = this.environments.get(id);
    if (!existing) {
      throw new EnvironmentNotFoundError(id);
    }

    const updated: Environment = {
      ...existing,
      name:
        input.name !== undefined
          ? validateEnvironmentName(input.name)
          : existing.name,
      slug:
        input.slug !== undefined
          ? validateEnvironmentSlug(input.slug)
          : existing.slug,
      description:
        input.description !== undefined
          ? input.description.trim() || undefined
          : existing.description,
      kind: input.kind ?? existing.kind,
      presentation:
        input.presentation !== undefined
          ? input.presentation
          : existing.presentation,
      status: input.status ?? existing.status,
      updatedAt: new Date().toISOString(),
    };
    this.environments.set(id, updated);
    return updated;
  }

  async setStatus(
    id: string,
    status: EnvironmentStatus,
  ): Promise<Environment> {
    return this.update(id, { status });
  }

  async delete(id: string): Promise<void> {
    if (!this.environments.has(id)) {
      throw new EnvironmentNotFoundError(id);
    }
    this.environments.delete(id);
  }
}
