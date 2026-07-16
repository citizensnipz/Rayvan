import { createHash, randomBytes } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, unlinkSync } from "node:fs";
import { join } from "node:path";

import {
  OsKeyringCredentialStore,
  type SecureCredentialStore,
} from "@rayvan/daemon-client";

/**
 * Daemon-owned OS-keyring store. The directory is retained only to migrate
 * legacy plaintext development blobs on first access.
 */
export class DaemonSecretStore {
  constructor(
    private readonly directory: string,
    private readonly secureStore: SecureCredentialStore = new OsKeyringCredentialStore(
      "com.rayvan.configuration-secret",
    ),
  ) {
    mkdirSync(directory, { recursive: true, mode: 0o700 });
  }

  put(value: string): { ref: string; fingerprint: string } {
    const ref = `sec_${randomBytes(16).toString("hex")}`;
    this.secureStore.set(ref, value);
    return { ref, fingerprint: fingerprint(value) };
  }

  get(ref: string): string | null {
    const stored = this.secureStore.get(ref);
    if (stored !== null) {
      return stored;
    }

    // One-time migration from the pre-keyring development format.
    const path = join(this.directory, `${ref}.bin`);
    if (!existsSync(path)) {
      return null;
    }
    const legacyValue = readFileSync(path, "utf8");
    this.secureStore.set(ref, legacyValue);
    unlinkSync(path);
    return legacyValue;
  }

  delete(ref: string): void {
    this.secureStore.delete(ref);
    const path = join(this.directory, `${ref}.bin`);
    if (existsSync(path)) {
      unlinkSync(path);
    }
  }
}

function fingerprint(value: string): string {
  return createHash("sha256").update(value).digest("hex").slice(0, 16);
}
