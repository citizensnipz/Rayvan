import type { PluginCapability } from "../manifest/index.js";

export type PluginExecutionStatus =
  | "succeeded"
  | "failed"
  | "cancelled"
  | "timed_out";

export type PluginExecutionErrorCode =
  | "not_found"
  | "capability_unsupported"
  | "missing_handler"
  | "validation_failed"
  | "permission_denied"
  | "approval_invalid"
  | "timeout"
  | "cancelled"
  | "execution_failed";

export interface SerializedPluginExecutionError {
  code: PluginExecutionErrorCode;
  message: string;
  pluginId: string;
  capability: PluginCapability;
  retryable: boolean;
  details?: Record<string, unknown>;
}

export interface PluginExecutionResultBase {
  executionId: string;
  pluginId: string;
  pluginVersion: string;
  capability: PluginCapability;
  status: PluginExecutionStatus;
  startedAt: string;
  finishedAt: string;
  durationMs: number;
  warnings: string[];
}

export type PluginExecutionResult<T> =
  | (PluginExecutionResultBase & {
      status: "succeeded";
      data: T;
      error?: undefined;
    })
  | (PluginExecutionResultBase & {
      status: "failed" | "cancelled" | "timed_out";
      data?: undefined;
      error: SerializedPluginExecutionError;
    });
