import type {
  AppliedConfigurationState,
  ConfigurationKey,
  ConfigurationOccurrence,
  ConfigurationValueAccess,
  DesiredConfigurationValue,
} from "@rayvan/core";

const DEFAULT_STALE_AFTER_MS = 7 * 24 * 60 * 60 * 1000;

export function normalizeComparableValue(
  value: string | undefined,
  valueType: ConfigurationKey["valueType"],
): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  const trimmed = value.trim();
  if (valueType === "boolean") {
    const lower = trimmed.toLowerCase();
    if (lower === "true" || lower === "1" || lower === "yes") {
      return "true";
    }
    if (lower === "false" || lower === "0" || lower === "no") {
      return "false";
    }
  }
  if (valueType === "number") {
    const asNumber = Number(trimmed);
    if (!Number.isNaN(asNumber)) {
      return String(asNumber);
    }
  }
  if (valueType === "json") {
    try {
      return JSON.stringify(JSON.parse(trimmed));
    } catch {
      return trimmed;
    }
  }
  if (valueType === "url") {
    try {
      const url = new URL(trimmed);
      return url.href.replace(/\/$/, "");
    } catch {
      return trimmed;
    }
  }
  return trimmed;
}

export function isComparableAccess(
  access: ConfigurationValueAccess,
): boolean {
  return access === "readable" || access === "masked";
}

export function isLockedAccess(access: ConfigurationValueAccess): boolean {
  return access === "locked" || access === "name_only";
}

/**
 * Compare desired vs observed for a single occurrence.
 * Sensitive: fingerprint only. Locked/name_only: never claim equality.
 */
export function desiredMatchesOccurrence(
  key: ConfigurationKey,
  desired: DesiredConfigurationValue,
  occurrence: ConfigurationOccurrence,
): "match" | "mismatch" | "locked" | "missing" | "unknown" {
  if (occurrence.valueAccess === "missing") {
    return "missing";
  }
  if (isLockedAccess(occurrence.valueAccess)) {
    return "locked";
  }
  if (!isComparableAccess(occurrence.valueAccess)) {
    return "unknown";
  }

  const sensitive = key.sensitive || key.valueType === "secret";
  if (sensitive) {
    if (!desired.valueFingerprint || !occurrence.valueFingerprint) {
      return "unknown";
    }
    return desired.valueFingerprint === occurrence.valueFingerprint
      ? "match"
      : "mismatch";
  }

  const desiredNorm = normalizeComparableValue(
    desired.desiredValue,
    key.valueType,
  );
  const observedNorm = normalizeComparableValue(
    occurrence.observedValue,
    key.valueType,
  );

  if (desiredNorm === undefined || observedNorm === undefined) {
    // Fall back to fingerprints when readable value is absent.
    if (desired.valueFingerprint && occurrence.valueFingerprint) {
      return desired.valueFingerprint === occurrence.valueFingerprint
        ? "match"
        : "mismatch";
    }
    return "unknown";
  }

  return desiredNorm === observedNorm ? "match" : "mismatch";
}

export function desiredMatchesApplied(
  desired: DesiredConfigurationValue,
  applied: AppliedConfigurationState | undefined,
): boolean {
  if (!applied) {
    return false;
  }
  if (applied.status === "failed" || applied.status === "verification_failed") {
    return false;
  }
  if (applied.desiredRevision !== desired.revision) {
    return false;
  }
  if (desired.valueFingerprint && applied.appliedFingerprint) {
    return desired.valueFingerprint === applied.appliedFingerprint;
  }
  return applied.desiredRevision === desired.revision;
}

export function appliedMatchesOccurrence(
  key: ConfigurationKey,
  applied: AppliedConfigurationState,
  occurrence: ConfigurationOccurrence,
): "match" | "mismatch" | "locked" | "unknown" {
  if (isLockedAccess(occurrence.valueAccess)) {
    return "locked";
  }
  if (!isComparableAccess(occurrence.valueAccess)) {
    return "unknown";
  }
  if (!applied.appliedFingerprint || !occurrence.valueFingerprint) {
    if (!key.sensitive && occurrence.observedValue !== undefined) {
      return "unknown";
    }
    return "unknown";
  }
  return applied.appliedFingerprint === occurrence.valueFingerprint
    ? "match"
    : "mismatch";
}

export function isObservedStale(
  occurrences: ConfigurationOccurrence[],
  nowIso: string,
  staleAfterMs = DEFAULT_STALE_AFTER_MS,
): boolean {
  if (occurrences.length === 0) {
    return false;
  }
  const now = Date.parse(nowIso);
  if (Number.isNaN(now)) {
    return false;
  }
  return occurrences.some((occurrence) => {
    const observedAt = Date.parse(occurrence.lastObservedAt);
    if (Number.isNaN(observedAt)) {
      return true;
    }
    return now - observedAt > staleAfterMs;
  });
}

export { DEFAULT_STALE_AFTER_MS };
