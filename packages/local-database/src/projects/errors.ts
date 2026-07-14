import { RayvanError } from "@rayvan/core";

export class ProjectNotFoundError extends RayvanError {
  constructor(id: string) {
    super("not_found", `Project not found: ${id}`);
    this.name = "ProjectNotFoundError";
  }
}

export class InvalidProjectNameError extends RayvanError {
  constructor() {
    super("validation_failed", "Project name must not be empty");
    this.name = "InvalidProjectNameError";
  }
}

export class ProjectPersistenceError extends RayvanError {
  constructor(message: string, cause?: unknown) {
    super("internal", message);
    this.name = "ProjectPersistenceError";
    if (cause instanceof Error) {
      this.cause = cause;
    }
  }
}
