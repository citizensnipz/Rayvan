import {
  DuplicateEnvironmentNameError,
  DuplicateEnvironmentSlugError,
  InvalidEnvironmentNameError,
} from "./errors.js";
import type { EnvironmentRepository } from "./repository.js";

const SLUG_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

export function validateEnvironmentName(name: string): string {
  const trimmed = name.trim();
  if (trimmed.length === 0) {
    throw new InvalidEnvironmentNameError();
  }
  return trimmed;
}

export function slugifyEnvironmentName(name: string): string {
  const slug = name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/-{2,}/g, "-");

  if (slug.length === 0) {
    throw new InvalidEnvironmentNameError(
      "Environment name must produce a non-empty slug",
    );
  }

  return slug;
}

export function validateEnvironmentSlug(slug: string): string {
  const trimmed = slug.trim().toLowerCase();
  if (trimmed.length === 0 || !SLUG_PATTERN.test(trimmed)) {
    throw new InvalidEnvironmentNameError(
      "Environment slug must be lowercase alphanumeric with hyphens",
    );
  }
  return trimmed;
}

export async function assertUniqueEnvironmentName(
  repository: EnvironmentRepository,
  projectId: string,
  name: string,
  excludeId?: string,
): Promise<void> {
  const existing = await repository.getByProjectAndName(projectId, name);
  if (existing && existing.id !== excludeId) {
    throw new DuplicateEnvironmentNameError(name);
  }
}

export async function assertUniqueEnvironmentSlug(
  repository: EnvironmentRepository,
  projectId: string,
  slug: string,
  excludeId?: string,
): Promise<void> {
  const existing = await repository.getByProjectAndSlug(projectId, slug);
  if (existing && existing.id !== excludeId) {
    throw new DuplicateEnvironmentSlugError(slug);
  }
}
