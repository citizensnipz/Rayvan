import { isAbsolute, normalize, resolve, sep } from "node:path";

import { PluginPackageError } from "./errors.js";

/**
 * Normalize and validate a zip entry name before extraction.
 * Rejects absolute paths, drive roots, UNC, and `..` segments.
 */
export function assertSafeZipEntryName(name: string): string {
  if (typeof name !== "string" || name.length === 0) {
    throw new PluginPackageError("Zip entry name is empty", "LAYOUT_INVALID");
  }
  if (name.includes("\0")) {
    throw new PluginPackageError(
      `Zip entry name contains NUL: ${name}`,
      "LAYOUT_INVALID",
    );
  }

  const normalized = name.replace(/\\/g, "/");
  if (
    normalized.startsWith("/") ||
    normalized.startsWith("//") ||
    /^[a-zA-Z]:/.test(normalized)
  ) {
    throw new PluginPackageError(
      `Zip entry name must be a relative path: ${name}`,
      "LAYOUT_INVALID",
    );
  }

  const parts = normalized.split("/");
  if (parts.some((part) => part === "" || part === "." || part === "..")) {
    throw new PluginPackageError(
      `Zip entry name contains unsafe path segment: ${name}`,
      "LAYOUT_INVALID",
    );
  }

  return normalized;
}

/**
 * Resolve `rel` under `rootDir` and ensure the result stays inside the root.
 */
export function resolveSafeInstallPath(rootDir: string, rel: string): string {
  const safeRel = assertSafeZipEntryName(rel);
  const resolvedRoot = resolve(rootDir);
  const resolvedDest = resolve(resolvedRoot, ...safeRel.split("/"));
  const rootWithSep = resolvedRoot.endsWith(sep)
    ? resolvedRoot
    : `${resolvedRoot}${sep}`;
  if (
    resolvedDest !== resolvedRoot &&
    !resolvedDest.startsWith(rootWithSep) &&
    !normalize(resolvedDest).startsWith(normalize(rootWithSep))
  ) {
    throw new PluginPackageError(
      `Zip entry escapes install root: ${rel}`,
      "LAYOUT_INVALID",
    );
  }
  if (isAbsolute(safeRel)) {
    throw new PluginPackageError(
      `Zip entry resolved to absolute path: ${rel}`,
      "LAYOUT_INVALID",
    );
  }
  return resolvedDest;
}
