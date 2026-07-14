/**
 * Tauri's generate_context! macro requires build.frontendDist to exist at
 * compile time. CI and fresh checkouts often lack apps/desktop/dist (gitignored).
 * Creating the directory is enough for cargo check/test; real bundles still use
 * beforeBuildCommand.
 */
import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "../..");
mkdirSync(join(root, "apps/desktop/dist"), { recursive: true });
