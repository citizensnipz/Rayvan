export class PluginPersistenceError extends Error {
  readonly cause?: unknown;

  constructor(message: string, cause?: unknown) {
    super(message);
    this.name = "PluginPersistenceError";
    this.cause = cause;
  }
}

export class PluginNotFoundError extends Error {
  constructor(readonly id: string) {
    super(`Plugin record not found: ${id}`);
    this.name = "PluginNotFoundError";
  }
}

export class PluginConnectionNotFoundError extends Error {
  constructor(readonly id: string) {
    super(`Plugin connection not found: ${id}`);
    this.name = "PluginConnectionNotFoundError";
  }
}

export class PluginDomainError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PluginDomainError";
  }
}

export class OptimisticConcurrencyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "OptimisticConcurrencyError";
  }
}
