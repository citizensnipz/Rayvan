import type {
  AppliedConfigurationStateId,
  ConfigurationKeyId,
  EnvironmentId,
  ProjectId,
} from "../ids/index.js";

export type AppliedConfigurationStatus =
  | "applied"
  | "verified"
  | "failed"
  | "verification_failed";

/**
 * Last-applied state for one
 * (configurationKeyId × environmentId × resourceBindingId).
 */
export interface AppliedConfigurationState {
  id: AppliedConfigurationStateId;
  configurationKeyId: ConfigurationKeyId;
  environmentId: EnvironmentId;
  projectId: ProjectId;
  resourceBindingId: string;

  /** Desired revision that was applied. */
  desiredRevision: number;
  /** Fingerprint of the value written (never plaintext secrets). */
  appliedFingerprint?: string;

  applyExecutionId: string;
  verificationExecutionId?: string;

  status: AppliedConfigurationStatus;
  appliedAt: string;
  verifiedAt?: string;
}
