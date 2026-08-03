import { generateKeyPairSync } from "node:crypto";
import { mkdtempSync, writeFileSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { zipSync } from "fflate";

import { RAYVAN_PLUGIN_API_VERSION, type PluginManifest } from "@rayvan/plugin-sdk";
import { describe, expect, it } from "vitest";

import { packPlugin } from "../src/pack.js";
import { installPluginPackageFromPath } from "../src/install.js";
import { PluginPackageError, verifyPluginPackage } from "../src/verify.js";
import { UNSIGNED_TRUST_LABEL } from "../src/trust.js";
import { detectHostTargetTriple } from "../src/triples.js";
import { resolveAllowUnsignedPlugins } from "../src/allow-unsigned.js";

const hostTriple = detectHostTargetTriple();

const manifest: PluginManifest = {
  id: "io.rayvan.github",
  name: "GitHub",
  version: "0.1.0",
  publisher: "rayvan",
  rayvanApiVersion: RAYVAN_PLUGIN_API_VERSION,
  capabilities: ["discover"],
  permissions: ["network"],
  resourceTypes: [
    {
      id: "github.repository",
      name: "Repository",
      schemaVersion: "1.0.0",
    },
  ],
  entrypoints: [
    {
      runtime: "native",
      binary: "rayvan-plugin-github",
      protocolVersion: "1",
    },
  ],
};

describe("@rayvan/plugin-package", () => {
  it("packs, verifies unsigned packages, and installs layout", () => {
    const dir = mkdtempSync(join(tmpdir(), "rayvan-plugin-pack-"));
    const binaryPath = join(dir, "rayvan-plugin-github.exe");
    writeFileSync(binaryPath, "#!/usr/bin/env node\nconsole.log('ok')\n");

    const packed = packPlugin({
      manifest,
      binaryPath,
      targetTriple: hostTriple,
      outputDir: dir,
    });
    expect(packed.signed).toBe(false);
    expect(packed.fileName).toBe(
      `io.rayvan.github-0.1.0-${hostTriple}.rayvan-plugin`,
    );

    const verified = verifyPluginPackage(packed.path, {
      trustedPublicKeys: [],
      allowUnsignedPlugins: true,
    });
    expect(verified.trustStatus).toBe("unsigned_development");
    expect(verified.trustLabel).toBe(UNSIGNED_TRUST_LABEL);
    expect(verified.manifest.id).toBe("io.rayvan.github");

    const installRoot = join(dir, "installed");
    const installed = installPluginPackageFromPath({
      packagePath: packed.path,
      installRoot,
      trustedPublicKeys: [],
      allowUnsignedPlugins: true,
    });
    expect(readFileSync(installed.manifestPath, "utf8")).toContain(
      "io.rayvan.github",
    );
    expect(installed.binaryPath.replace(/\\/g, "/")).toContain(
      "bin/rayvan-plugin-github",
    );
  });

  it("hard-rejects invalid signatures and unsigned when gated", () => {
    const dir = mkdtempSync(join(tmpdir(), "rayvan-plugin-pack-"));
    const binaryPath = join(dir, "rayvan-plugin-github");
    writeFileSync(binaryPath, "binary");

    const { privateKey, publicKey } = generateKeyPairSync("ed25519");
    const packed = packPlugin({
      manifest,
      binaryPath,
      targetTriple: hostTriple,
      outputDir: dir,
      signingPrivateKeyPem: privateKey.export({ type: "pkcs8", format: "pem" }).toString(),
    });
    expect(packed.signed).toBe(true);

    const ok = verifyPluginPackage(packed.path, {
      trustedPublicKeys: [
        publicKey.export({ type: "spki", format: "pem" }).toString(),
      ],
      allowUnsignedPlugins: false,
    });
    expect(ok.trustStatus).toBe("signed");

    const unsigned = packPlugin({
      manifest: { ...manifest, version: "0.1.1" },
      binaryPath,
      targetTriple: hostTriple,
      outputDir: dir,
    });
    expect(() =>
      verifyPluginPackage(unsigned.path, {
        trustedPublicKeys: [],
        allowUnsignedPlugins: false,
      }),
    ).toThrow(PluginPackageError);

    const other = generateKeyPairSync("ed25519");
    expect(() =>
      verifyPluginPackage(packed.path, {
        trustedPublicKeys: [
          other.publicKey.export({ type: "spki", format: "pem" }).toString(),
        ],
        allowUnsignedPlugins: false,
      }),
    ).toThrow(/Signature verification failed/);
  });

  it("rejects zip path traversal entries", () => {
    const dir = mkdtempSync(join(tmpdir(), "rayvan-plugin-zip-"));
    const evil = zipSync({
      "manifest.json": Buffer.from("{}"),
      "SHA256SUMS": Buffer.from("deadbeef  manifest.json\n"),
      "bin/ok": Buffer.from("x"),
      "../escape.txt": Buffer.from("pwned"),
    });
    const packagePath = join(dir, `evil-0.0.1-${hostTriple}.rayvan-plugin`);
    writeFileSync(packagePath, evil);
    expect(() =>
      verifyPluginPackage(packagePath, {
        trustedPublicKeys: [],
        allowUnsignedPlugins: true,
      }),
    ).toThrow(/unsafe path segment|relative path/i);
  });

  it("rejects absolute zip entry names", () => {
    const dir = mkdtempSync(join(tmpdir(), "rayvan-plugin-zip-"));
    const evil = zipSync({
      "/tmp/evil": Buffer.from("x"),
      "manifest.json": Buffer.from("{}"),
      "SHA256SUMS": Buffer.from("x"),
      "bin/ok": Buffer.from("x"),
    });
    const packagePath = join(dir, `evil-abs-0.0.1-${hostTriple}.rayvan-plugin`);
    writeFileSync(packagePath, evil);
    expect(() =>
      verifyPluginPackage(packagePath, {
        trustedPublicKeys: [],
        allowUnsignedPlugins: true,
      }),
    ).toThrow(/relative path/i);
  });

  it("rejects host triple mismatch on install", () => {
    const dir = mkdtempSync(join(tmpdir(), "rayvan-plugin-pack-"));
    const binaryPath = join(dir, "bin");
    writeFileSync(binaryPath, "binary");
    const otherTriple =
      hostTriple === "x86_64-unknown-linux-gnu"
        ? "x86_64-pc-windows-msvc"
        : "x86_64-unknown-linux-gnu";
    const packed = packPlugin({
      manifest,
      binaryPath,
      targetTriple: otherTriple,
      outputDir: dir,
    });
    expect(() =>
      installPluginPackageFromPath({
        packagePath: packed.path,
        installRoot: join(dir, "installed"),
        trustedPublicKeys: [],
        allowUnsignedPlugins: true,
      }),
    ).toThrow(/does not match host/);
  });

  it("resolveAllowUnsignedPlugins is opt-in only", () => {
    expect(resolveAllowUnsignedPlugins(undefined, undefined)).toBe(false);
    expect(resolveAllowUnsignedPlugins(undefined, "0")).toBe(false);
    expect(resolveAllowUnsignedPlugins(undefined, "1")).toBe(true);
    expect(resolveAllowUnsignedPlugins(undefined, "TRUE")).toBe(true);
    expect(resolveAllowUnsignedPlugins(true, "0")).toBe(true);
    expect(resolveAllowUnsignedPlugins(false, "1")).toBe(false);
  });
});
