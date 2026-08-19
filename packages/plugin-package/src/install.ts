import { chmodSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

import type { PluginManifest } from "@rayvan/plugin-sdk";

import { PluginPackageError } from "./errors.js";
import { resolveSafeInstallPath } from "./paths.js";
import {
  detectHostTargetTriple,
  type PluginTargetTriple,
} from "./triples.js";
import type { PluginTrustStatus } from "./trust.js";
import {
  verifyPluginPackage,
  type VerifiedPluginPackage,
} from "./verify.js";
import type { VerifySignatureOptions } from "./trust.js";

export interface InstalledPluginPackageLayout {
  rootDir: string;
  manifestPath: string;
  binaryPath: string;
  manifest: PluginManifest;
  trustStatus: PluginTrustStatus;
  trustLabel: string;
  signerFingerprint?: string;
  targetTriple?: PluginTargetTriple;
  packagePath: string;
}

export interface InstallPluginPackageOptions extends VerifySignatureOptions {
  packagePath: string;
  /** Destination root, typically `<dataDir>/plugins/<pluginId>/<version>/<triple>`. */
  installRoot: string;
  /**
   * When set, reject packages whose filename triple does not match.
   * Defaults to the detected host triple when `enforceHostTriple` is true.
   */
  expectedTargetTriple?: PluginTargetTriple;
  /** When true (default), require package triple to match the host. */
  enforceHostTriple?: boolean;
}

function resolveBinaryPath(
  files: ReadonlyMap<string, Uint8Array>,
  rootDir: string,
): string {
  const binEntries = [...files.keys()].filter((name) =>
    name.startsWith("bin/"),
  );
  if (binEntries.length === 0) {
    throw new PluginPackageError("Package has no bin/ entries", "LAYOUT_INVALID");
  }
  // Prefer a non-directory single binary; take the first sorted entry.
  binEntries.sort();
  return resolveSafeInstallPath(rootDir, binEntries[0]!);
}

/**
 * Verify a `.rayvan-plugin` archive and extract it into `installRoot`.
 * Does not mutate Rayvan SQLite — callers persist InstalledPluginRecord.
 */
export function installPluginPackageFromPath(
  options: InstallPluginPackageOptions,
): InstalledPluginPackageLayout {
  const verified: VerifiedPluginPackage = verifyPluginPackage(
    options.packagePath,
    {
      trustedPublicKeys: options.trustedPublicKeys,
      allowUnsignedPlugins: options.allowUnsignedPlugins,
    },
  );

  const enforceHostTriple = options.enforceHostTriple !== false;
  if (enforceHostTriple) {
    const expected =
      options.expectedTargetTriple ?? detectHostTargetTriple();
    if (!verified.targetTriple) {
      throw new PluginPackageError(
        "Plugin package filename is missing a supported target triple",
        "HOST_MISMATCH",
      );
    }
    if (verified.targetTriple !== expected) {
      throw new PluginPackageError(
        `Plugin package target ${verified.targetTriple} does not match host ${expected}`,
        "HOST_MISMATCH",
      );
    }
  }

  mkdirSync(options.installRoot, { recursive: true });

  for (const [rel, content] of verified.files) {
    const dest = resolveSafeInstallPath(options.installRoot, rel);
    mkdirSync(dirname(dest), { recursive: true });
    writeFileSync(dest, content);
    if (rel.startsWith("bin/") && !rel.includes("windows")) {
      try {
        chmodSync(dest, 0o755);
      } catch {
        // Windows / non-POSIX: ignore
      }
    }
  }

  return {
    rootDir: options.installRoot,
    manifestPath: resolveSafeInstallPath(options.installRoot, "manifest.json"),
    binaryPath: resolveBinaryPath(verified.files, options.installRoot),
    manifest: verified.manifest,
    trustStatus: verified.trustStatus,
    trustLabel: verified.trustLabel,
    signerFingerprint: verified.signerFingerprint,
    targetTriple: verified.targetTriple,
    packagePath: options.packagePath,
  };
}
