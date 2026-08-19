/**
 * Resolve whether unsigned plugin packages are permitted.
 *
 * Opt-in only: explicit `true`, or env `RAYVAN_ALLOW_UNSIGNED_PLUGINS` set to
 * `1` / `true` (case-insensitive). Default is deny.
 */
export function resolveAllowUnsignedPlugins(
  explicit?: boolean,
  envValue: string | undefined = process.env.RAYVAN_ALLOW_UNSIGNED_PLUGINS,
): boolean {
  if (explicit !== undefined) {
    return explicit;
  }
  const normalized = envValue?.trim().toLowerCase();
  return normalized === "1" || normalized === "true";
}
