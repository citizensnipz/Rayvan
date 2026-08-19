import { createPrivateKey, createPublicKey, sign, verify } from "node:crypto";

import { SIGNATURE_NAME, sha256Hex } from "./checksums.js";

export type PluginTrustStatus =
  | "signed"
  | "unsigned_development"
  | "rejected";

export const UNSIGNED_TRUST_LABEL = "Unsigned (development)";

export interface VerifySignatureOptions {
  /** Ed25519 public keys (raw 32-byte or SPKI PEM) trusted by the host. */
  trustedPublicKeys: readonly (Buffer | Uint8Array | string)[];
  /** When true, packages without SIGNATURE.ed25519 may be accepted. */
  allowUnsignedPlugins: boolean;
}

export interface TrustVerificationResult {
  status: PluginTrustStatus;
  label: string;
  signerFingerprint?: string;
}

function normalizePublicKey(key: Buffer | Uint8Array | string) {
  if (typeof key === "string") {
    return createPublicKey(key);
  }
  const buf = Buffer.isBuffer(key) ? key : Buffer.from(key);
  if (buf.length === 32) {
    return createPublicKey({
      key: Buffer.concat([
        Buffer.from("302a300506032b6570032100", "hex"),
        buf,
      ]),
      format: "der",
      type: "spki",
    });
  }
  return createPublicKey({ key: buf, format: "der", type: "spki" });
}

/** Sign SHA256SUMS bytes with an Ed25519 private key (PEM or raw 64-byte seed+pub). */
export function signSha256Sums(
  sha256SumsBytes: Buffer | Uint8Array,
  privateKeyPemOrDer: string | Buffer,
): Buffer {
  const key =
    typeof privateKeyPemOrDer === "string"
      ? createPrivateKey(privateKeyPemOrDer)
      : createPrivateKey({
          key: privateKeyPemOrDer,
          format: "der",
          type: "pkcs8",
        });
  return sign(null, Buffer.from(sha256SumsBytes), key);
}

export function verifyPackageTrust(input: {
  files: ReadonlyMap<string, Uint8Array | Buffer>;
  sha256SumsBytes: Buffer | Uint8Array;
  options: VerifySignatureOptions;
}): TrustVerificationResult {
  const signature = input.files.get(SIGNATURE_NAME);
  if (!signature) {
    if (input.options.allowUnsignedPlugins) {
      return {
        status: "unsigned_development",
        label: UNSIGNED_TRUST_LABEL,
      };
    }
    return {
      status: "rejected",
      label: "Missing signature (unsigned plugins not allowed)",
    };
  }

  const sums = Buffer.from(input.sha256SumsBytes);
  for (const publicKey of input.options.trustedPublicKeys) {
    try {
      const keyObject = normalizePublicKey(publicKey);
      const ok = verify(null, sums, keyObject, Buffer.from(signature));
      if (ok) {
        const exported = keyObject.export({ type: "spki", format: "der" });
        return {
          status: "signed",
          label: "Signed",
          signerFingerprint: sha256Hex(exported).slice(0, 16),
        };
      }
    } catch {
      // try next key
    }
  }

  return {
    status: "rejected",
    label: "Signature verification failed",
  };
}
