import type {
  AuthenticateContext,
  ApplyContext,
  DiscoveryContext,
  EvaluateFindingsContext,
  InspectContext,
  PlanContext,
  VerifyContext,
} from "./contexts/index.js";
import type {
  ApplyResult,
  AuthenticateResult,
  ChangePlan,
  DiscoveredResource,
  EvaluateFindingsResult,
  ObservedResourceState,
  VerificationResult,
} from "./contracts/index.js";
import type { PluginManifest } from "./manifest/index.js";

export type AuthenticateHandler = (
  context: AuthenticateContext,
) => Promise<AuthenticateResult>;

export type DiscoverHandler = (
  context: DiscoveryContext,
) => Promise<DiscoveredResource[]>;

export type InspectHandler = (
  context: InspectContext,
) => Promise<ObservedResourceState>;

export type PlanHandler = (context: PlanContext) => Promise<ChangePlan>;

export type ApplyHandler = (context: ApplyContext) => Promise<ApplyResult>;

export type VerifyHandler = (
  context: VerifyContext,
) => Promise<VerificationResult>;

export type EvaluateFindingsHandler = (
  context: EvaluateFindingsContext,
) => Promise<EvaluateFindingsResult>;

/**
 * Plugin contract with optional lifecycle handlers.
 * Declare a capability in the manifest only when the matching handler exists.
 * `evaluateFindings` returns detections only — never write Finding records.
 */
export interface RayvanPlugin {
  manifest: PluginManifest;

  authenticate?: AuthenticateHandler;
  discover?: DiscoverHandler;
  inspect?: InspectHandler;
  plan?: PlanHandler;
  apply?: ApplyHandler;
  verify?: VerifyHandler;
  evaluateFindings?: EvaluateFindingsHandler;
}
