import { RayvanError } from "@rayvan/core";

export class FindingDomainError extends RayvanError {
  constructor(
    code: "not_found" | "validation_failed" | "conflict" | "internal",
    message: string,
  ) {
    super(code, message);
    this.name = "FindingDomainError";
  }
}

export class FindingNotFoundError extends FindingDomainError {
  constructor(id: string) {
    super("not_found", `Finding not found: ${id}`);
    this.name = "FindingNotFoundError";
  }
}

export class InvalidFindingStatusTransitionError extends FindingDomainError {
  constructor(message: string) {
    super("validation_failed", message);
    this.name = "InvalidFindingStatusTransitionError";
  }
}

export class FindingPersistenceError extends FindingDomainError {
  constructor(message: string, cause?: unknown) {
    super("internal", message);
    this.name = "FindingPersistenceError";
    if (cause instanceof Error) {
      this.cause = cause;
    }
  }
}
