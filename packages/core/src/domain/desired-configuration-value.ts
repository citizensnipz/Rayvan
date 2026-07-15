import type {
  ConfigurationKeyId,
  DesiredConfigurationValueId,
  EnvironmentId,
  ProjectId,
} from "../ids/index.js";

/**
 * Who last updated a desired configuration value.
 * Mirrors plugin actor shape with `kind` for persistence-friendly JSON.
 */
export type ConfigurationActorRef =
  | { kind: "user"; id: string; displayName?: string }
  | { kind: "system"; id: string }
  | { kind: "mcp_agent"; id: string; displayName?: string };

/**
 * Desired configuration value for one (configurationKeyId × environmentId).
 *
 * Sensitive keys store only `secretValueRef` + `valueFingerprint` —
 * never plaintext in `desiredValue`.
 */
export interface DesiredConfigurationValue {
  id: DesiredConfigurationValueId;
  configurationKeyId: ConfigurationKeyId;
  environmentId: EnvironmentId;
  projectId: ProjectId;

  /** Non-sensitive desired plaintext only. */
  desiredValue?: string;
  /** Secure-store reference for sensitive values. */
  secretValueRef?: string;
  /** Fingerprint for comparison without exposing secrets. */
  valueFingerprint?: string;

  revision: number;
  createdAt: string;
  updatedAt: string;
  updatedBy: ConfigurationActorRef;
}
