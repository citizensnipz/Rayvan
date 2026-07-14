const SENSITIVE_KEY_PATTERN =
  /^(token|secret|password|authorization|apiKey|accessToken|refreshToken)$/i;

const REDACTED = "[REDACTED]";

/**
 * Deep-clone a value while redacting known secret-bearing keys
 * (case-insensitive). Used on warnings, errors, and event payloads.
 * Cycle-safe: circular references become "[Circular]".
 */
export function redactSecrets<T>(value: T): T {
  try {
    return redactValue(value, new WeakSet<object>()) as T;
  } catch {
    return "[REDACTED]" as T;
  }
}

function redactValue(value: unknown, seen: WeakSet<object>): unknown {
  if (value === null || value === undefined) {
    return value;
  }

  if (Array.isArray(value)) {
    if (seen.has(value)) {
      return "[Circular]";
    }
    seen.add(value);
    return value.map((item) => redactValue(item, seen));
  }

  if (typeof value === "object") {
    if (seen.has(value)) {
      return "[Circular]";
    }
    seen.add(value);

    if (value instanceof Error) {
      return {
        name: value.name,
        message: value.message,
        ...(value.cause !== undefined
          ? { cause: redactValue(value.cause, seen) }
          : {}),
      };
    }

    const result: Record<string, unknown> = {};
    for (const [key, nested] of Object.entries(
      value as Record<string, unknown>,
    )) {
      result[key] = SENSITIVE_KEY_PATTERN.test(key)
        ? REDACTED
        : redactValue(nested, seen);
    }
    return result;
  }

  return value;
}
