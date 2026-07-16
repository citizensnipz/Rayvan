import type { DaemonErrorCode, DaemonSerializedError } from "@rayvan/daemon-contracts";

export class DaemonAppError extends Error {
  readonly code: DaemonErrorCode;
  readonly retryable: boolean;
  readonly details?: Record<string, string | number | boolean | null>;
  readonly correlationId?: string;

  constructor(
    code: DaemonErrorCode,
    message: string,
    options?: {
      retryable?: boolean;
      details?: Record<string, string | number | boolean | null>;
      correlationId?: string;
      cause?: unknown;
    },
  ) {
    super(message, options?.cause ? { cause: options.cause } : undefined);
    this.name = "DaemonAppError";
    this.code = code;
    this.retryable = options?.retryable ?? false;
    this.details = options?.details;
    this.correlationId = options?.correlationId;
  }

  toSerialized(): DaemonSerializedError {
    return {
      code: this.code,
      message: this.message,
      retryable: this.retryable,
      correlationId: this.correlationId,
      details: this.details,
    };
  }
}

export function toDaemonError(
  error: unknown,
  correlationId?: string,
): DaemonSerializedError {
  if (error instanceof DaemonAppError) {
    return error.toSerialized();
  }
  if (error instanceof Error) {
    const code = (error as Error & { code?: string }).code;
    if (typeof code === "string" && code.includes("REVISION")) {
      return {
        code: "REVISION_CONFLICT",
        message: error.message,
        retryable: true,
        correlationId,
      };
    }
    return {
      code: "INTERNAL_ERROR",
      message: "Internal daemon error",
      retryable: false,
      correlationId,
    };
  }
  return {
    code: "INTERNAL_ERROR",
    message: "Unknown error",
    retryable: false,
    correlationId,
  };
}
