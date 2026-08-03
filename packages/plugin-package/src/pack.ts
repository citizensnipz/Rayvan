import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { basename, join } from "node:path";
import { zipSync } from "fflate";

import {
  validatePluginManifest,
  type PluginManifest,
} from "@rayvan/plugin-sdk";

import {
  MANIFEST_NAME,
  SHA256SUMS_NAME,
  SIGNATURE_NAME,
  buildSha256Sums,
} from "./checksums.js";
import { signSha256Sums } from "./trust.js";
import {
  binaryFileName,
  packageFileName,
  type PluginTargetTriple,
} from "./triples.js";

export interface PackPluginOptions {
  manifest: PluginManifest;
  /** Absolute or relative path to the native launcher binary. */
  binaryPath: string;
  targetTriple: PluginTargetTriple;
  outputDir: string;
  /** Optional asset files keyed by path under `assets/`. */
  assets?: ReadonlyMap<string, Uint8Array | Buffer>;
  /** Optional Ed25519 private key PEM; omit for unsigned/dev packages. */
  signingPrivateKeyPem?: string;
}

export interface PackedPluginArtifact {
  path: string;
  fileName: string;
  files: Map<string, Uint8Array>;
  signed: boolean;
}

export function packPlugin(options: PackPluginOptions): PackedPluginArtifact {
  validatePluginManifest(options.manifest);

  const entry = options.manifest.entrypoints?.find(
    (item) => item.runtime === "native",
  );
  const binaryBase = entry?.binary ?? `rayvan-plugin-${options.manifest.id.split(".").pop()}`;
  const binName = binaryFileName(binaryBase, options.targetTriple);
  const binRel = `bin/${binName}`;

  if (!existsSync(options.binaryPath)) {
    throw new Error(`Binary not found: ${options.binaryPath}`);
  }

  const files = new Map<string, Uint8Array>();
  const manifestJson = `${JSON.stringify(options.manifest, null, 2)}\n`;
  files.set(MANIFEST_NAME, Buffer.from(manifestJson, "utf8"));
  files.set(binRel, readFileSync(options.binaryPath));

  if (options.assets) {
    for (const [rel, content] of options.assets) {
      const normalized = rel.replace(/\\/g, "/").replace(/^\/+/, "");
      files.set(`assets/${normalized}`, Buffer.from(content));
    }
  }

  const sums = buildSha256Sums(files);
  files.set(SHA256SUMS_NAME, Buffer.from(sums, "utf8"));

  let signed = false;
  if (options.signingPrivateKeyPem) {
    const signature = signSha256Sums(
      Buffer.from(sums, "utf8"),
      options.signingPrivateKeyPem,
    );
    files.set(SIGNATURE_NAME, signature);
    signed = true;
  }

  const zipEntries: Record<string, Uint8Array> = {};
  for (const [name, content] of files) {
    zipEntries[name] = content;
  }
  const zipped = zipSync(zipEntries, { level: 6 });

  mkdirSync(options.outputDir, { recursive: true });
  const fileName = packageFileName(
    options.manifest.id,
    options.manifest.version,
    options.targetTriple,
  );
  const outPath = join(options.outputDir, fileName);
  writeFileSync(outPath, zipped);

  return {
    path: outPath,
    fileName: basename(outPath),
    files,
    signed,
  };
}
