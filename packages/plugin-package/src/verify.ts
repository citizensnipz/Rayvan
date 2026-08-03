import { readFileSync, statSync } from "node:fs";
import { unzipSync } from "fflate";

import {
  validatePluginManifest,
  type PluginManifest,
} from "@rayvan/plugin-sdk";

import {
  MANIFEST_NAME,
  SHA256SUMS_NAME,
  buildSha256Sums,
  parseSha256Sums,
  sha256Hex,
} from "./checksums.js";
import { PluginPackageError } from "./errors.js";
import { PLUGIN_PACKAGE_LIMITS } from "./limits.js";
import { assertSafeZipEntryName } from "./paths.js";
import {
  UNSIGNED_TRUST_LABEL,
  verifyPackageTrust,
  type PluginTrustStatus,
  type VerifySignatureOptions,
} from "./trust.js";
import { isPluginTargetTriple, type PluginTargetTriple } from "./triples.js";

export interface VerifiedPluginPackage {
  manifest: PluginManifest;
  files: Map<string, Uint8Array>;
  trustStatus: PluginTrustStatus;
  trustLabel: string;
  signerFingerprint?: string;
  targetTriple?: PluginTargetTriple;
  packagePath: string;
}

export { PluginPackageError } from "./errors.js";

export function readPluginPackageArchive(
  packagePath: string,
): Map<string, Uint8Array> {
  let archiveBytes: Buffer;
  try {
    const stats = statSync(packagePath);
    if (stats.size > PLUGIN_PACKAGE_LIMITS.maxArchiveBytes) {
      throw new PluginPackageError(
        `Plugin package exceeds max archive size (${PLUGIN_PACKAGE_LIMITS.maxArchiveBytes} bytes)`,
        "INVALID_ARCHIVE",
      );
    }
    archiveBytes = readFileSync(packagePath);
  } catch (error) {
    if (error instanceof PluginPackageError) throw error;
    throw new PluginPackageError(
      `Failed to read plugin package: ${error instanceof Error ? error.message : String(error)}`,
      "INVALID_ARCHIVE",
    );
  }

  let unzipped: Record<string, Uint8Array>;
  try {
    unzipped = unzipSync(archiveBytes);
  } catch (error) {
    throw new PluginPackageError(
      `Failed to unzip plugin package: ${error instanceof Error ? error.message : String(error)}`,
      "INVALID_ARCHIVE",
    );
  }

  const files = new Map<string, Uint8Array>();
  let totalUncompressed = 0;
  for (const [name, content] of Object.entries(unzipped)) {
    const raw = name.replace(/\\/g, "/");
    if (raw.endsWith("/")) continue;
    const normalized = assertSafeZipEntryName(raw);
    if (content.byteLength > PLUGIN_PACKAGE_LIMITS.maxEntryBytes) {
      throw new PluginPackageError(
        `Zip entry exceeds max size: ${normalized}`,
        "INVALID_ARCHIVE",
      );
    }
    totalUncompressed += content.byteLength;
    if (totalUncompressed > PLUGIN_PACKAGE_LIMITS.maxTotalUncompressedBytes) {
      throw new PluginPackageError(
        "Plugin package exceeds max uncompressed size",
        "INVALID_ARCHIVE",
      );
    }
    files.set(normalized, content);
    if (files.size > PLUGIN_PACKAGE_LIMITS.maxEntries) {
      throw new PluginPackageError(
        `Plugin package exceeds max entry count (${PLUGIN_PACKAGE_LIMITS.maxEntries})`,
        "INVALID_ARCHIVE",
      );
    }
  }
  return files;
}

export function verifyPluginPackage(
  packagePath: string,
  options: VerifySignatureOptions,
): VerifiedPluginPackage {
  const files = readPluginPackageArchive(packagePath);

  if (!files.has(MANIFEST_NAME)) {
    throw new PluginPackageError("manifest.json missing", "LAYOUT_INVALID");
  }
  if (!files.has(SHA256SUMS_NAME)) {
    throw new PluginPackageError("SHA256SUMS missing", "LAYOUT_INVALID");
  }

  const hasBin = [...files.keys()].some((name) => name.startsWith("bin/"));
  if (!hasBin) {
    throw new PluginPackageError("bin/ entry missing", "LAYOUT_INVALID");
  }

  // Verify checksums against payload files (excludes SHA256SUMS + SIGNATURE).
  const sumsBytes = files.get(SHA256SUMS_NAME)!;
  const expected = parseSha256Sums(Buffer.from(sumsBytes).toString("utf8"));
  const recomputed = parseSha256Sums(buildSha256Sums(files));

  if (expected.size !== recomputed.size) {
    throw new PluginPackageError(
      "SHA256SUMS entry count mismatch",
      "CHECKSUM_MISMATCH",
    );
  }
  for (const [name, hash] of expected) {
    assertSafeZipEntryName(name);
    const actual = recomputed.get(name);
    if (!actual || actual !== hash.toLowerCase()) {
      throw new PluginPackageError(
        `Checksum mismatch for ${name}`,
        "CHECKSUM_MISMATCH",
      );
    }
    const content = files.get(name);
    if (!content || sha256Hex(content) !== hash.toLowerCase()) {
      throw new PluginPackageError(
        `Checksum mismatch for ${name}`,
        "CHECKSUM_MISMATCH",
      );
    }
  }

  let manifest: PluginManifest;
  try {
    manifest = JSON.parse(
      Buffer.from(files.get(MANIFEST_NAME)!).toString("utf8"),
    ) as PluginManifest;
    validatePluginManifest(manifest);
  } catch (error) {
    throw new PluginPackageError(
      error instanceof Error ? error.message : "Invalid manifest",
      "MANIFEST_INVALID",
    );
  }

  const trust = verifyPackageTrust({
    files,
    sha256SumsBytes: sumsBytes,
    options,
  });
  if (trust.status === "rejected") {
    throw new PluginPackageError(trust.label, "SIGNATURE_REJECTED");
  }

  const base = packagePath.replace(/\\/g, "/").split("/").pop() ?? "";
  const tripleMatch = /-([a-z0-9_]+(?:-[a-z0-9_]+)+)\.rayvan-plugin$/i.exec(
    base,
  );
  const triple = tripleMatch?.[1];
  const targetTriple =
    triple && isPluginTargetTriple(triple) ? triple : undefined;

  return {
    manifest,
    files,
    trustStatus: trust.status,
    trustLabel:
      trust.status === "unsigned_development"
        ? UNSIGNED_TRUST_LABEL
        : trust.label,
    signerFingerprint: trust.signerFingerprint,
    targetTriple,
    packagePath,
  };
}
