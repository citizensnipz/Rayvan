import type {
  Environment,
  EnvironmentKind,
  EnvironmentPresentation,
  EnvironmentStatus,
} from "@rayvan/core";

import { EnvironmentNotFoundError } from "./errors.js";
import type { EnvironmentRepository } from "./repository.js";
import {
  assertUniqueEnvironmentName,
  assertUniqueEnvironmentSlug,
  slugifyEnvironmentName,
  validateEnvironmentName,
  validateEnvironmentSlug,
} from "./validation.js";

function defaultPresentation(kind: EnvironmentKind): EnvironmentPresentation {
  switch (kind) {
    case "local":
      return { color: "neutral", icon: "local" };
    case "development":
      return { color: "blue", icon: "development" };
    case "preview":
      return { color: "violet", icon: "preview" };
    case "staging":
      return { color: "amber", icon: "staging" };
    case "production":
      return { color: "rose", icon: "production" };
    case "test":
      return { color: "cyan", icon: "test" };
    case "custom":
      return { color: "neutral", icon: "custom" };
  }
}

export interface CreateEnvironmentServiceInput {
  projectId: string;
  name: string;
  slug?: string;
  kind: EnvironmentKind;
  description?: string;
  presentation?: EnvironmentPresentation;
  status?: EnvironmentStatus;
}

export interface UpdateEnvironmentServiceInput {
  name?: string;
  slug?: string;
  description?: string;
  kind?: EnvironmentKind;
  presentation?: EnvironmentPresentation;
  status?: EnvironmentStatus;
}

export class EnvironmentService {
  constructor(private readonly repository: EnvironmentRepository) {}

  list(
    projectId: string,
    options?: { includeArchived?: boolean },
  ): Promise<Environment[]> {
    return this.repository.listByProjectId(projectId, options);
  }

  getById(id: string): Promise<Environment | null> {
    return this.repository.getById(id);
  }

  async create(input: CreateEnvironmentServiceInput): Promise<Environment> {
    const name = validateEnvironmentName(input.name);
    const slug = validateEnvironmentSlug(
      input.slug ?? slugifyEnvironmentName(name),
    );

    await assertUniqueEnvironmentName(
      this.repository,
      input.projectId,
      name,
    );
    await assertUniqueEnvironmentSlug(
      this.repository,
      input.projectId,
      slug,
    );

    return this.repository.create({
      projectId: input.projectId,
      name,
      slug,
      kind: input.kind,
      description: input.description,
      presentation: input.presentation ?? defaultPresentation(input.kind),
      status: input.status ?? "local_only",
    });
  }

  async update(
    id: string,
    input: UpdateEnvironmentServiceInput,
  ): Promise<Environment> {
    const existing = await this.repository.getById(id);
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
        : input.name !== undefined && input.slug === undefined
          ? validateEnvironmentSlug(slugifyEnvironmentName(name))
          : existing.slug;

    if (name !== existing.name) {
      await assertUniqueEnvironmentName(
        this.repository,
        existing.projectId,
        name,
        id,
      );
    }
    if (slug !== existing.slug) {
      await assertUniqueEnvironmentSlug(
        this.repository,
        existing.projectId,
        slug,
        id,
      );
    }

    const kind = input.kind ?? existing.kind;
    const presentation =
      input.presentation !== undefined
        ? input.presentation
        : input.kind !== undefined && input.presentation === undefined
          ? defaultPresentation(kind)
          : existing.presentation;

    return this.repository.update(id, {
      name,
      slug,
      description: input.description,
      kind,
      presentation,
      status: input.status,
    });
  }

  async archive(id: string): Promise<Environment> {
    const existing = await this.repository.getById(id);
    if (!existing) {
      throw new EnvironmentNotFoundError(id);
    }
    return this.repository.setStatus(id, "archived");
  }

  async restore(id: string): Promise<Environment> {
    const existing = await this.repository.getById(id);
    if (!existing) {
      throw new EnvironmentNotFoundError(id);
    }
    return this.repository.setStatus(id, "local_only");
  }
}
