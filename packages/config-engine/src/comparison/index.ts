import type { ConfigurationEntry, ConfigurationSnapshot } from "@rayvan/core";

export interface ConfigurationComparisonResult {
  missingKeys: string[];
  extraKeys: string[];
  driftedKeys: string[];
}

export function compareConfigurationSnapshots(
  baseline: ConfigurationSnapshot,
  target: ConfigurationSnapshot,
): ConfigurationComparisonResult {
  const baselineKeys = new Map(
    baseline.entries.map((entry) => [entry.key, entry.valueFingerprint]),
  );
  const targetKeys = new Map(
    target.entries.map((entry) => [entry.key, entry.valueFingerprint]),
  );

  const missingKeys: string[] = [];
  const extraKeys: string[] = [];
  const driftedKeys: string[] = [];

  for (const key of baselineKeys.keys()) {
    if (!targetKeys.has(key)) {
      missingKeys.push(key);
      continue;
    }
    if (baselineKeys.get(key) !== targetKeys.get(key)) {
      driftedKeys.push(key);
    }
  }

  for (const key of targetKeys.keys()) {
    if (!baselineKeys.has(key)) {
      extraKeys.push(key);
    }
  }

  return { missingKeys, extraKeys, driftedKeys };
}

export function findMissingRequiredEntries(
  entries: ConfigurationEntry[],
): ConfigurationEntry[] {
  return entries.filter((entry) => entry.isRequired && !entry.valueFingerprint);
}
