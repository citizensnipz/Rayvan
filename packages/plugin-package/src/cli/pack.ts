#!/usr/bin/env node
/**
 * Thin pack helper:
 *   node dist/cli/pack.js --manifest <path> --binary <path> --triple <triple> --out <dir>
 */
import { readFileSync } from "node:fs";

import type { PluginManifest } from "@rayvan/plugin-sdk";

import { packPlugin } from "../pack.js";
import { isPluginTargetTriple } from "../triples.js";

function arg(name: string): string | undefined {
  const idx = process.argv.indexOf(name);
  if (idx < 0) return undefined;
  return process.argv[idx + 1];
}

function main(): void {
  const manifestPath = arg("--manifest");
  const binaryPath = arg("--binary");
  const triple = arg("--triple");
  const out = arg("--out") ?? "dist/plugins";
  const keyPath = arg("--signing-key");

  if (!manifestPath || !binaryPath || !triple) {
    console.error(
      "Usage: rayvan-plugin-pack --manifest <manifest.json> --binary <path> --triple <target-triple> [--out <dir>] [--signing-key <pem>]",
    );
    process.exit(1);
  }
  if (!isPluginTargetTriple(triple)) {
    console.error(`Unsupported target triple: ${triple}`);
    process.exit(1);
  }

  const manifest = JSON.parse(
    readFileSync(manifestPath, "utf8"),
  ) as PluginManifest;
  const artifact = packPlugin({
    manifest,
    binaryPath,
    targetTriple: triple,
    outputDir: out,
    signingPrivateKeyPem: keyPath
      ? readFileSync(keyPath, "utf8")
      : undefined,
  });
  console.log(JSON.stringify({ path: artifact.path, signed: artifact.signed }));
}

main();
