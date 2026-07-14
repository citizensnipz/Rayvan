import type { PluginCapability } from "../manifest/index.js";

export type PluginErrorCode =
  | "validation_failed"
  | "not_found"
  | "capability_unsupported"
  | "execution_failed"
  | "version_mismatch";

export class PluginError extends Error {
  readonly code: PluginErrorCode;
  readonly pluginId?: string;

  constructor(
    code: PluginErrorCode,
    message: string,
    options?: { pluginId?: string; cause?: unknown },
  ) {
    super(message, options?.cause !== undefined ? { cause: options.cause } : undefined);
    this.name = "PluginError";
    this.code = code;
    this.pluginId = options?.pluginId;
  }
}

export class PluginValidationError extends PluginError {
  constructor(message: string, options?: { pluginId?: string; cause?: unknown }) {
    super("validation_failed", message, options);
    this.name = "PluginValidationError";
  }
}

export class PluginNotFoundError extends PluginError {
  constructor(pluginId: string) {
    super("not_found", `Plugin not found: ${pluginId}`, { pluginId });
    this.name = "PluginNotFoundError";
  }
}

export class PluginCapabilityError extends PluginError {
  readonly capability: PluginCapability;

  constructor(
    pluginId: string,
    capability: PluginCapability,
    message?: string,
  ) {
    super(
      "capability_unsupported",
      message ?? `Plugin "${pluginId}" does not support capability "${capability}"`,
      { pluginId },
    );
    this.name = "PluginCapabilityError";
    this.capability = capability;
  }
}

export class PluginExecutionError extends PluginError {
  readonly capability: PluginCapability;

  constructor(
    pluginId: string,
    capability: PluginCapability,
    message: string,
    options?: { cause?: unknown },
  ) {
    super("execution_failed", message, { pluginId, cause: options?.cause });
    this.name = "PluginExecutionError";
    this.capability = capability;
  }
}

export class PluginVersionError extends PluginError {
  constructor(message: string, options?: { pluginId?: string; cause?: unknown }) {
    super("version_mismatch", message, options);
    this.name = "PluginVersionError";
  }
}
