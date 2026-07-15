import { RayvanError } from "@rayvan/core";

export class EnvironmentDomainError extends RayvanError {
  constructor(code: "not_found" | "validation_failed" | "conflict" | "internal", message: string) {
    super(code, message);
    this.name = "EnvironmentDomainError";
  }
}

export class EnvironmentNotFoundError extends EnvironmentDomainError {
  constructor(id: string) {
    super("not_found", `Environment not found: ${id}`);
    this.name = "EnvironmentNotFoundError";
  }
}

export class InvalidEnvironmentNameError extends EnvironmentDomainError {
  constructor(message = "Environment name must not be empty") {
    super("validation_failed", message);
    this.name = "InvalidEnvironmentNameError";
  }
}

export class DuplicateEnvironmentNameError extends EnvironmentDomainError {
  constructor(name: string) {
    super("conflict", `Environment name already exists in project: ${name}`);
    this.name = "DuplicateEnvironmentNameError";
  }
}

export class DuplicateEnvironmentSlugError extends EnvironmentDomainError {
  constructor(slug: string) {
    super("conflict", `Environment slug already exists in project: ${slug}`);
    this.name = "DuplicateEnvironmentSlugError";
  }
}

export class EnvironmentPersistenceError extends EnvironmentDomainError {
  constructor(message: string, cause?: unknown) {
    super("internal", message);
    this.name = "EnvironmentPersistenceError";
    if (cause instanceof Error) {
      this.cause = cause;
    }
  }
}
