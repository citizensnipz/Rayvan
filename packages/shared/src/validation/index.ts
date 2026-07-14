export function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

export function assertNonEmptyString(
  value: unknown,
  field: string,
): asserts value is string {
  if (!isNonEmptyString(value)) {
    throw new Error(`${field} must be a non-empty string`);
  }
}
