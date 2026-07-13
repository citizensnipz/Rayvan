export const REDACTED_VALUE = "[REDACTED]";

export function redactSecretValue(isSecret: boolean, value?: string): string | undefined {
  if (!value) {
    return value;
  }
  return isSecret ? REDACTED_VALUE : value;
}

export function redactConfigurationKey(key: string): string {
  return key;
}
