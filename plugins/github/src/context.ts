import type {
  ApplyContext,
  AuthenticateContext,
  DiscoveryContext,
  InspectContext,
  PlanContext,
  VerifyContext,
} from "@rayvan/plugin-sdk";

import { GithubClient } from "./client.js";

type MediatedContext = {
  credentials?: { accessToken?: string };
  connectionMetadata?: Record<string, unknown>;
};

export function isFixtureContext(context: MediatedContext): boolean {
  return (
    context.connectionMetadata?.fixture === true ||
    process.env.RAYVAN_GITHUB_FIXTURE === "1"
  );
}

export function createClientFromContext(context: MediatedContext): GithubClient {
  const fixture = isFixtureContext(context);
  const token = context.credentials?.accessToken;
  return new GithubClient({
    fixture,
    credentials: token
      ? { token, tokenType: "bearer" }
      : undefined,
  });
}

export type GithubCapabilityContext =
  | AuthenticateContext
  | DiscoveryContext
  | InspectContext
  | PlanContext
  | ApplyContext
  | VerifyContext;
