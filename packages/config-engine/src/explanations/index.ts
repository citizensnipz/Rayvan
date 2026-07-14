import type { ConfigurationEntry } from "@rayvan/core";

export function explainConfigurationEntry(entry: ConfigurationEntry): string {
  if (entry.description) {
    return entry.description;
  }
  if (entry.isSecret) {
    return `Secret configuration value for ${entry.key}.`;
  }
  return `Configuration value for ${entry.key}.`;
}
