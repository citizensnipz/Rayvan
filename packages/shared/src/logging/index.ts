export type LogLevel = "debug" | "info" | "warn" | "error";

export interface Logger {
  debug(message: string, context?: Record<string, unknown>): void;
  info(message: string, context?: Record<string, unknown>): void;
  warn(message: string, context?: Record<string, unknown>): void;
  error(message: string, context?: Record<string, unknown>): void;
}

export function createConsoleLogger(scope: string): Logger {
  const prefix = `[${scope}]`;
  return {
    debug: (message, context) => console.debug(prefix, message, context ?? ""),
    info: (message, context) => console.info(prefix, message, context ?? ""),
    warn: (message, context) => console.warn(prefix, message, context ?? ""),
    error: (message, context) => console.error(prefix, message, context ?? ""),
  };
}
