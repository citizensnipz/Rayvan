import type {
  ConfigurationKey,
  ConfigurationOccurrence,
  DesiredConfigurationValue,
} from "@rayvan/core";

import {
  InvalidConfigurationKeyNameError,
  PlaintextSecretNotAllowedError,
} from "./errors.js";

export function validateConfigurationKeyName(name: string): string {
  const trimmed = name.trim();
  if (trimmed.length === 0) {
    throw new InvalidConfigurationKeyNameError();
  }
  return trimmed;
}

/**
 * Reject plaintext secrets in ordinary occurrence storage.
 * Sensitive readable values may only use secretValueRef (and optional maskedValue / fingerprint).
 */
export function assertNoPlaintextSecret(input: {
  sensitive: boolean;
  observedValue?: string;
}): void {
  if (!input.sensitive) {
    return;
  }

  if (input.observedValue !== undefined && input.observedValue !== "") {
    throw new PlaintextSecretNotAllowedError();
  }
}

export function assertNoPlaintextSecretForKey(
  key: ConfigurationKey,
  occurrence: Pick<ConfigurationOccurrence, "observedValue">,
): void {
  assertNoPlaintextSecret({
    sensitive: key.sensitive || key.valueType === "secret",
    observedValue: occurrence.observedValue,
  });
}

/**
 * Reject plaintext desired values for sensitive keys.
 * Sensitive desired state may only use secretValueRef (+ optional fingerprint).
 */
export function assertNoPlaintextSecretDesired(input: {
  sensitive: boolean;
  desiredValue?: string;
}): void {
  if (!input.sensitive) {
    return;
  }

  if (input.desiredValue !== undefined && input.desiredValue !== "") {
    throw new PlaintextSecretNotAllowedError();
  }
}

export function assertNoPlaintextSecretDesiredForKey(
  key: ConfigurationKey,
  desired: Pick<DesiredConfigurationValue, "desiredValue">,
): void {
  assertNoPlaintextSecretDesired({
    sensitive: key.sensitive || key.valueType === "secret",
    desiredValue: desired.desiredValue,
  });
}
