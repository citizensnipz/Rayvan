import { FINDING_FINGERPRINT_VERSION } from "@rayvan/core";

import { sha256Hex } from "./sha256.js";

/**
 * Build a stable Finding fingerprint from rule + project + structural parts.
 * Titles and timestamps must never contribute — wording changes must not fork identity.
 *
 * Uses a pure JS SHA-256 so this package can be bundled into the desktop app
 * without importing Node's `node:crypto`.
 */
export function buildFindingFingerprint(input: {
  ruleId: string;
  projectId: string;
  fingerprintParts: readonly string[];
}): string {
  const parts = input.fingerprintParts
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
  const canonical = [
    FINDING_FINGERPRINT_VERSION,
    input.ruleId,
    input.projectId,
    ...parts,
  ].join("\u0000");
  return sha256Hex(canonical);
}

export function fingerprintVersion(): string {
  return FINDING_FINGERPRINT_VERSION;
}
