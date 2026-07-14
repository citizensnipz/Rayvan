import { PluginDomainError } from "./errors.js";

const SECRET_KEY_PATTERN =
  /(password|secret|token|api[_-]?key|private[_-]?key|credential)/i;

/**
 * Rejects plaintext secret-like string values at any nesting depth.
 * Allowed patterns: credential reference ids, hashes, masked values, booleans.
 */
export function assertNoPlaintextSecrets(
  value: unknown,
  path = "state",
): void {
  if (value === null || value === undefined) {
    return;
  }
  if (Array.isArray(value)) {
    for (let index = 0; index < value.length; index += 1) {
      assertNoPlaintextSecrets(value[index], `${path}[${index}]`);
    }
    return;
  }
  if (typeof value !== "object") {
    return;
  }

  for (const [key, nested] of Object.entries(
    value as Record<string, unknown>,
  )) {
    const nestedPath = `${path}.${key}`;
    if (SECRET_KEY_PATTERN.test(key) && typeof nested === "string") {
      const trimmed = nested.trim();
      const looksMasked =
        trimmed.length === 0 ||
        trimmed === "[redacted]" ||
        trimmed.startsWith("***") ||
        trimmed.startsWith("cred:") ||
        trimmed.startsWith("hash:");
      if (!looksMasked) {
        throw new PluginDomainError(
          `Plaintext secret values are not allowed in persisted state (${nestedPath}); use a credential reference or metadata`,
        );
      }
    }
    assertNoPlaintextSecrets(nested, nestedPath);
  }
}
