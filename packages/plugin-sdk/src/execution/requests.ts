import type {
  ApplyContext,
  AuthenticateContext,
  DiscoveryContext,
  EvaluateFindingsContext,
  InspectContext,
  PlanContext,
  VerifyContext,
} from "../contexts/index.js";
import type { PluginExecutionActor } from "./actor.js";

export type { PluginExecutionActor } from "./actor.js";

export interface PluginExecutionRequestBase {
  pluginId: string;
  projectId?: string;
  environmentId?: string;
  resourceId?: string;
  actor: PluginExecutionActor;
  reason?: string;
  timeoutMs?: number;
  signal?: AbortSignal;
}

export interface AuthenticateExecutionRequest extends PluginExecutionRequestBase {
  context: AuthenticateContext;
}

export interface DiscoverExecutionRequest extends PluginExecutionRequestBase {
  context: DiscoveryContext;
}

export interface InspectExecutionRequest extends PluginExecutionRequestBase {
  context: InspectContext;
}

export interface PlanExecutionRequest extends PluginExecutionRequestBase {
  context: PlanContext;
}

export interface ApplyExecutionRequest extends PluginExecutionRequestBase {
  context: ApplyContext;
}

export interface VerifyExecutionRequest extends PluginExecutionRequestBase {
  context: VerifyContext;
}

export interface EvaluateFindingsExecutionRequest
  extends PluginExecutionRequestBase {
  context: EvaluateFindingsContext;
}

export type PluginExecutionRequest =
  | AuthenticateExecutionRequest
  | DiscoverExecutionRequest
  | InspectExecutionRequest
  | PlanExecutionRequest
  | ApplyExecutionRequest
  | VerifyExecutionRequest
  | EvaluateFindingsExecutionRequest;
