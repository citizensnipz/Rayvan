/**
 * Emit dist/manifest.json for packing after `pnpm build`.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const distManifestJs = join(here, "../dist/manifest.js");
const { manifest } = await import(pathToFileURL(distManifestJs).href);
const outDir = join(here, "../dist");
mkdirSync(outDir, { recursive: true });
writeFileSync(
  join(outDir, "manifest.json"),
  `${JSON.stringify(manifest, null, 2)}\n`,
);
console.log("Wrote dist/manifest.json");
