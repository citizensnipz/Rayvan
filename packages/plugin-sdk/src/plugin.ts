import type {
  AuthenticateContext,
  ApplyContext,
  DiscoveryContext,
  InspectContext,
  PlanContext,
  VerifyContext,
} from "./contexts/index.js";
import type {
  ApplyResult,
  AuthenticateResult,
  ChangePlan,
  DiscoveredResource,
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

/**
 * Plugin contract with optional lifecycle handlers.
 * Declare a capability in the manifest only when the matching handler exists.
 */
export interface RayvanPlugin {
  manifest: PluginManifest;

  authenticate?: AuthenticateHandler;
  discover?: DiscoverHandler;
  inspect?: InspectHandler;
  plan?: PlanHandler;
  apply?: ApplyHandler;
  verify?: VerifyHandler;
}
