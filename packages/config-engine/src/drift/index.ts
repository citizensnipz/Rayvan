import type { ConfigurationEntry } from "@rayvan/core";

export interface DriftFinding {
  key: string;
  baselineFingerprint?: string;
  targetFingerprint?: string;
}

export function detectFingerprintDrift(
  baseline: ConfigurationEntry[],
  target: ConfigurationEntry[],
): DriftFinding[] {
  const baselineByKey = new Map(baseline.map((entry) => [entry.key, entry]));
  const findings: DriftFinding[] = [];

  for (const [key, baselineEntry] of baselineByKey) {
    const targetEntry = target.find((entry) => entry.key === key);
    if (!targetEntry) {
      continue;
    }
    if (baselineEntry.valueFingerprint !== targetEntry.valueFingerprint) {
      findings.push({
        key,
        baselineFingerprint: baselineEntry.valueFingerprint,
        targetFingerprint: targetEntry.valueFingerprint,
      });
    }
  }

  return findings;
}
