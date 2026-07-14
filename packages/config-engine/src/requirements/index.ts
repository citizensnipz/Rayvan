import type { ConfigurationEntry } from "@rayvan/core";

export function listUnmetRequirements(entries: ConfigurationEntry[]): string[] {
  return entries
    .filter((entry) => entry.isRequired && !entry.valueFingerprint)
    .map((entry) => entry.key);
}
