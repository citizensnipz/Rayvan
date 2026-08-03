import { createHash } from "node:crypto";

export const SHA256SUMS_NAME = "SHA256SUMS";
export const SIGNATURE_NAME = "SIGNATURE.ed25519";
export const MANIFEST_NAME = "manifest.json";

/**
 * Files excluded from SHA256SUMS coverage.
 * SHA256SUMS cannot checksum itself; SIGNATURE covers the SUMS bytes.
 */
export const CHECKSUM_EXCLUDED = new Set([SHA256SUMS_NAME, SIGNATURE_NAME]);

export function sha256Hex(data: Uint8Array | Buffer | string): string {
  return createHash("sha256").update(data).digest("hex");
}

/**
 * Build a GNU-style SHA256SUMS body covering every package file except
 * SIGNATURE.ed25519. Paths use forward slashes.
 */
export function buildSha256Sums(
  files: ReadonlyMap<string, Uint8Array | Buffer>,
): string {
  const lines: string[] = [];
  const names = [...files.keys()]
    .filter((name) => !CHECKSUM_EXCLUDED.has(name))
    .sort();
  for (const name of names) {
    const content = files.get(name)!;
    lines.push(`${sha256Hex(content)}  ${name}`);
  }
  return `${lines.join("\n")}\n`;
}

export function parseSha256Sums(
  body: string,
): Map<string, string> {
  const result = new Map<string, string>();
  for (const rawLine of body.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const match = /^([a-f0-9]{64})\s+(\S+)$/i.exec(line);
    if (!match) {
      throw new Error(`Invalid SHA256SUMS line: ${rawLine}`);
    }
    result.set(match[2]!, match[1]!.toLowerCase());
  }
  return result;
}
