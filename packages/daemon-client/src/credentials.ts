import { createHash, randomBytes, timingSafeEqual } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { dirname } from "node:path";

import { Entry } from "@napi-rs/keyring";

import { daemonCredentialStorePath } from "./paths.js";

interface StoredCredential {
  clientId: string;
  /** SHA-256 hex digest of the raw credential. */
  credentialHash: string;
  createdAt: string;
  rotatedAt?: string;
}

interface CredentialFile {
  version: 1;
  credentials: StoredCredential[];
}

export interface SecureCredentialStore {
  set(account: string, value: string): void;
  get(account: string): string | null;
  delete(account: string): void;
}

export class OsKeyringCredentialStore implements SecureCredentialStore {
  constructor(private readonly service = "com.rayvan.local-client") {}

  set(account: string, value: string): void {
    new Entry(this.service, account).setPassword(value);
  }

  get(account: string): string | null {
    try {
      return new Entry(this.service, account).getPassword();
    } catch (error) {
      if (isMissingCredential(error)) {
        return null;
      }
      throw error;
    }
  }

  delete(account: string): void {
    try {
      new Entry(this.service, account).deleteCredential();
    } catch (error) {
      if (!isMissingCredential(error)) {
        throw error;
      }
    }
  }
}

/** Stores only credential hashes in the metadata file; raw tokens live in the OS keyring. */
export class LocalClientCredentialStore {
  constructor(
    private readonly path = daemonCredentialStorePath(),
    private readonly secureStore: SecureCredentialStore = new OsKeyringCredentialStore(),
  ) {}

  private read(): CredentialFile {
    if (!existsSync(this.path)) {
      return { version: 1, credentials: [] };
    }
    const raw = readFileSync(this.path, "utf8");
    return JSON.parse(raw) as CredentialFile;
  }

  private write(file: CredentialFile): void {
    mkdirSync(dirname(this.path), { recursive: true, mode: 0o700 });
    writeFileSync(this.path, JSON.stringify(file, null, 2), {
      encoding: "utf8",
      mode: 0o600,
    });
  }

  issue(clientId: string): string {
    const credential = `rvc_${randomBytes(32).toString("base64url")}`;
    const credentialHash = hashCredential(credential);
    const file = this.read();
    const previousCredential = this.secureStore.get(clientId);
    file.credentials = file.credentials.filter((c) => c.clientId !== clientId);
    file.credentials.push({
      clientId,
      credentialHash,
      createdAt: new Date().toISOString(),
    });
    this.secureStore.set(clientId, credential);
    try {
      this.write(file);
    } catch (error) {
      if (previousCredential === null) {
        this.secureStore.delete(clientId);
      } else {
        this.secureStore.set(clientId, previousCredential);
      }
      throw error;
    }
    return credential;
  }

  resolve(clientId: string): string | null {
    const credential = this.secureStore.get(clientId);
    if (credential) {
      return credential;
    }

    // One-time migration from the pre-keyring development format.
    const legacyTokenPath = `${dirname(this.path)}/${clientId}.token`;
    if (!existsSync(legacyTokenPath)) {
      return null;
    }
    const legacyCredential = readFileSync(legacyTokenPath, "utf8").trim();
    if (!legacyCredential || !this.verify(clientId, legacyCredential)) {
      unlinkSync(legacyTokenPath);
      return null;
    }
    this.secureStore.set(clientId, legacyCredential);
    unlinkSync(legacyTokenPath);
    return legacyCredential;
  }

  verify(clientId: string, credential: string): boolean {
    const file = this.read();
    const entry = file.credentials.find((c) => c.clientId === clientId);
    if (!entry) {
      return false;
    }
    const actual = Buffer.from(hashCredential(credential), "utf8");
    const expected = Buffer.from(entry.credentialHash, "utf8");
    if (actual.length !== expected.length) {
      return false;
    }
    return timingSafeEqual(actual, expected);
  }

  revoke(clientId: string): void {
    const file = this.read();
    file.credentials = file.credentials.filter((c) => c.clientId !== clientId);
    this.write(file);
    this.secureStore.delete(clientId);
    const legacyTokenPath = `${dirname(this.path)}/${clientId}.token`;
    if (existsSync(legacyTokenPath)) unlinkSync(legacyTokenPath);
  }

  rotate(clientId: string): string {
    return this.issue(clientId);
  }
}

function hashCredential(credential: string): string {
  return createHash("sha256").update(credential).digest("hex");
}

function isMissingCredential(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return /no entry|not found|does not exist/i.test(message);
}
