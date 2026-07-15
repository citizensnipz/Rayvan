import { RayvanError } from "@rayvan/core";

export class ConfigurationDomainError extends RayvanError {
  constructor(
    code: "not_found" | "validation_failed" | "conflict" | "internal",
    message: string,
  ) {
    super(code, message);
    this.name = "ConfigurationDomainError";
  }
}

export class ConfigurationKeyNotFoundError extends ConfigurationDomainError {
  constructor(id: string) {
    super("not_found", `Configuration key not found: ${id}`);
    this.name = "ConfigurationKeyNotFoundError";
  }
}

export class ConfigurationOccurrenceNotFoundError extends ConfigurationDomainError {
  constructor(id: string) {
    super("not_found", `Configuration occurrence not found: ${id}`);
    this.name = "ConfigurationOccurrenceNotFoundError";
  }
}

export class InvalidConfigurationKeyNameError extends ConfigurationDomainError {
  constructor(message = "Configuration key name must not be empty") {
    super("validation_failed", message);
    this.name = "InvalidConfigurationKeyNameError";
  }
}

export class PlaintextSecretNotAllowedError extends ConfigurationDomainError {
  constructor() {
    super(
      "validation_failed",
      "Sensitive configuration values must not store plaintext in observedValue; use secretValueRef",
    );
    this.name = "PlaintextSecretNotAllowedError";
  }
}

export class DesiredConfigurationValueNotFoundError extends ConfigurationDomainError {
  constructor(id: string) {
    super("not_found", `Desired configuration value not found: ${id}`);
    this.name = "DesiredConfigurationValueNotFoundError";
  }
}

export class DesiredConfigurationRevisionConflictError extends ConfigurationDomainError {
  constructor(expectedRevision: number, actualRevision: number) {
    super(
      "conflict",
      `Desired configuration revision mismatch: expected ${expectedRevision}, found ${actualRevision}`,
    );
    this.name = "DesiredConfigurationRevisionConflictError";
  }
}

export class DuplicateConfigurationKeyNameError extends ConfigurationDomainError {
  constructor(name: string) {
    super(
      "conflict",
      `Configuration key name already exists in project: ${name}`,
    );
    this.name = "DuplicateConfigurationKeyNameError";
  }
}

export class ConfigurationPersistenceError extends ConfigurationDomainError {
  constructor(message: string, cause?: unknown) {
    super("internal", message);
    this.name = "ConfigurationPersistenceError";
    if (cause instanceof Error) {
      this.cause = cause;
    }
  }
}
