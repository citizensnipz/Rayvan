import type { PluginCapability } from "../../manifest/index.js";
import type { PluginExecutionActor } from "../actor.js";
import type {
  PluginExecutionErrorCode,
  PluginExecutionStatus,
} from "../results.js";

export interface PluginExecutionEvent {
  executionId: string;
  pluginId: string;
  pluginVersion: string;
  capability: PluginCapability;
  status: PluginExecutionStatus;
  startedAt: string;
  finishedAt: string;
  durationMs: number;
  actor: PluginExecutionActor;
  projectId?: string;
  environmentId?: string;
  resourceId?: string;
  reason?: string;
  errorCode?: PluginExecutionErrorCode;
  errorMessage?: string;
  warningCount: number;
}

export interface PluginExecutionEventSink {
  record(event: PluginExecutionEvent): void | Promise<void>;
}
