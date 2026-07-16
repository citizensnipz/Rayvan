import type {
  FindingDetection,
  FindingEvidence,
  FindingObservedState,
  FindingRuleDefinition,
  FindingSeverity,
  SafeFindingValue,
} from "@rayvan/core";
import { isFindingCategory, isFindingSeverity } from "@rayvan/core";

export class FindingValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "FindingValidationError";
  }
}

/**
 * Obvious secret material in free-text evidence strings.
 * Duplicated (intentionally) from plugin-sdk — do not import plugin-sdk here.
 */
const OBVIOUS_SECRET_PATTERNS: readonly RegExp[] = [
  /\bBearer\s+[A-Za-z0-9\-._~+/]+=*/i,
  /\b(?:api[_-]?key|secret|password|token|access[_-]?token|refresh[_-]?token)\s*[:=]\s*\S+/i,
  /\bsk-(?:live|test)?[_-]?[A-Za-z0-9]{16,}\b/,
  /\bghp_[A-Za-z0-9]{20,}\b/,
  /\bgho_[A-Za-z0-9]{20,}\b/,
  /\bxox[baprs]-[A-Za-z0-9-]{10,}\b/i,
];

const SAFE_VALUE_ACCESSES = new Set([
  "readable",
  "fingerprint",
  "masked",
  "locked",
  "name_only",
  "unknown",
]);

function assertNoObviousSecrets(value: string, field: string): void {
  for (const pattern of OBVIOUS_SECRET_PATTERNS) {
    if (pattern.test(value)) {
      throw new FindingValidationError(
        `${field} appears to contain a secret and must be redacted`,
      );
    }
  }
}

function validateSafeFindingValue(
  value: SafeFindingValue,
  field: string,
): void {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new FindingValidationError(`${field} must be an object`);
  }
  if (!SAFE_VALUE_ACCESSES.has(value.access)) {
    throw new FindingValidationError(
      `${field}.access has unsupported value "${String((value as { access?: unknown }).access)}"`,
    );
  }
  if (typeof value.sensitive !== "boolean") {
    throw new FindingValidationError(`${field}.sensitive must be a boolean`);
  }
  if (value.access === "readable") {
    if (value.sensitive !== false) {
      throw new FindingValidationError(
        `${field}.sensitive must be false for readable values`,
      );
    }
    if (typeof value.value !== "string" || value.value.trim().length === 0) {
      throw new FindingValidationError(`${field}.value must be a non-empty string`);
    }
    assertNoObviousSecrets(value.value, `${field}.value`);
  } else if (value.access === "fingerprint") {
    if (value.sensitive !== true) {
      throw new FindingValidationError(
        `${field}.sensitive must be true for fingerprint values`,
      );
    }
    if (
      typeof value.fingerprint !== "string" ||
      value.fingerprint.trim().length === 0
    ) {
      throw new FindingValidationError(
        `${field}.fingerprint must be a non-empty string`,
      );
    }
  } else if (value.access === "masked") {
    if (value.sensitive !== true) {
      throw new FindingValidationError(
        `${field}.sensitive must be true for masked values`,
      );
    }
    if (value.maskedValue !== undefined) {
      if (
        typeof value.maskedValue !== "string" ||
        value.maskedValue.trim().length === 0
      ) {
        throw new FindingValidationError(
          `${field}.maskedValue must be a non-empty string when provided`,
        );
      }
      assertNoObviousSecrets(value.maskedValue, `${field}.maskedValue`);
    }
  }
}

function validateObservedState(
  state: FindingObservedState,
  field: string,
): void {
  validateSafeFindingValue(state.value, `${field}.value`);
}

function validateEvidenceItem(
  evidence: FindingEvidence,
  field: string,
): void {
  switch (evidence.type) {
    case "message":
      if (typeof evidence.message !== "string" || evidence.message.trim() === "") {
        throw new FindingValidationError(`${field}.message must be a non-empty string`);
      }
      assertNoObviousSecrets(evidence.message, `${field}.message`);
      break;
    case "connection_error":
      if (
        typeof evidence.safeMessage !== "string" ||
        evidence.safeMessage.trim() === ""
      ) {
        throw new FindingValidationError(
          `${field}.safeMessage must be a non-empty string`,
        );
      }
      assertNoObviousSecrets(evidence.safeMessage, `${field}.safeMessage`);
      break;
    case "resource_state":
      if (typeof evidence.state !== "string" || evidence.state.trim() === "") {
        throw new FindingValidationError(`${field}.state must be a non-empty string`);
      }
      assertNoObviousSecrets(evidence.state, `${field}.state`);
      break;
    case "deployment_state":
      if (typeof evidence.status !== "string" || evidence.status.trim() === "") {
        throw new FindingValidationError(`${field}.status must be a non-empty string`);
      }
      assertNoObviousSecrets(evidence.status, `${field}.status`);
      break;
    case "configuration_comparison":
      if (evidence.expectedState !== undefined) {
        validateSafeFindingValue(
          evidence.expectedState,
          `${field}.expectedState`,
        );
      }
      if (!Array.isArray(evidence.observedStates)) {
        throw new FindingValidationError(
          `${field}.observedStates must be an array`,
        );
      }
      for (let i = 0; i < evidence.observedStates.length; i += 1) {
        validateObservedState(
          evidence.observedStates[i]!,
          `${field}.observedStates[${i}]`,
        );
      }
      break;
    default:
      break;
  }
}

export function validateRuleDefinition(
  rule: FindingRuleDefinition,
): FindingRuleDefinition {
  if (!rule.id.trim()) {
    throw new FindingValidationError("Rule id is required");
  }
  if (!rule.name.trim()) {
    throw new FindingValidationError(`Rule ${rule.id}: name is required`);
  }
  if (!isFindingCategory(rule.category)) {
    throw new FindingValidationError(
      `Rule ${rule.id}: invalid category ${rule.category}`,
    );
  }
  if (!isFindingSeverity(rule.defaultSeverity)) {
    throw new FindingValidationError(
      `Rule ${rule.id}: invalid defaultSeverity ${rule.defaultSeverity}`,
    );
  }
  return rule;
}

export function validateDetection(
  detection: FindingDetection,
  knownRuleIds?: ReadonlySet<string>,
): FindingDetection {
  if (!detection.ruleId.trim()) {
    throw new FindingValidationError("Detection ruleId is required");
  }
  if (knownRuleIds && !knownRuleIds.has(detection.ruleId)) {
    throw new FindingValidationError(
      `Detection references unknown ruleId: ${detection.ruleId}`,
    );
  }
  if (!detection.projectId || String(detection.projectId).trim() === "") {
    throw new FindingValidationError(
      `Detection for ${detection.ruleId}: projectId is required`,
    );
  }
  if (!detection.title.trim()) {
    throw new FindingValidationError(
      `Detection for ${detection.ruleId}: title is required`,
    );
  }
  if (!detection.summary.trim()) {
    throw new FindingValidationError(
      `Detection for ${detection.ruleId}: summary is required`,
    );
  }
  if (!Array.isArray(detection.fingerprintParts)) {
    throw new FindingValidationError(
      `Detection for ${detection.ruleId}: fingerprintParts must be an array`,
    );
  }
  if (detection.fingerprintParts.length === 0) {
    throw new FindingValidationError(
      `Detection for ${detection.ruleId}: fingerprintParts must not be empty`,
    );
  }
  if (
    detection.severity !== undefined &&
    !isFindingSeverity(detection.severity)
  ) {
    throw new FindingValidationError(
      `Detection for ${detection.ruleId}: invalid severity ${detection.severity}`,
    );
  }
  if (!Array.isArray(detection.evidence)) {
    throw new FindingValidationError(
      `Detection for ${detection.ruleId}: evidence must be an array`,
    );
  }
  for (let i = 0; i < detection.evidence.length; i += 1) {
    validateEvidenceItem(
      detection.evidence[i]!,
      `Detection for ${detection.ruleId}: evidence[${i}]`,
    );
  }
  return detection;
}

export function resolveDetectionSeverity(
  detection: FindingDetection,
  defaultSeverity: FindingSeverity,
  override?: FindingSeverity,
): FindingSeverity {
  return override ?? detection.severity ?? defaultSeverity;
}
