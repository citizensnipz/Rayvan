import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  LocalClientCredentialStore,
  type SecureCredentialStore,
} from "../src/credentials.js";

class MemorySecureStore implements SecureCredentialStore {
  readonly values = new Map<string, string>();
  failNextSet = false;

  set(account: string, value: string): void {
    if (this.failNextSet) {
      this.failNextSet = false;
      throw new Error("keyring unavailable");
    }
    this.values.set(account, value);
  }

  get(account: string): string | null {
    return this.values.get(account) ?? null;
  }

  delete(account: string): void {
    this.values.delete(account);
  }
}

const roots: string[] = [];

afterEach(() => {
  for (const root of roots.splice(0)) {
    rmSync(root, { recursive: true, force: true });
  }
});

describe("LocalClientCredentialStore", () => {
  it("persists only a hash and resolves the raw token through secure storage", () => {
    const root = mkdtempSync(join(tmpdir(), "rayvan-credentials-"));
    roots.push(root);
    const metadataPath = join(root, "local-clients.json");
    const secureStore = new MemorySecureStore();
    const credentials = new LocalClientCredentialStore(metadataPath, secureStore);

    const token = credentials.issue("client-1");

    expect(credentials.verify("client-1", token)).toBe(true);
    expect(credentials.resolve("client-1")).toBe(token);
    expect(readFileSync(metadataPath, "utf8")).not.toContain(token);
    expect(existsSync(join(root, "client-1.token"))).toBe(false);
  });

  it("removes secure credentials when a client is revoked", () => {
    const root = mkdtempSync(join(tmpdir(), "rayvan-credentials-"));
    roots.push(root);
    const secureStore = new MemorySecureStore();
    const credentials = new LocalClientCredentialStore(
      join(root, "local-clients.json"),
      secureStore,
    );
    credentials.issue("client-1");

    credentials.revoke("client-1");

    expect(credentials.resolve("client-1")).toBeNull();
    expect(secureStore.values.has("client-1")).toBe(false);
  });

  it("treats orphan keyring secrets without metadata as unresolved", () => {
    const root = mkdtempSync(join(tmpdir(), "rayvan-credentials-"));
    roots.push(root);
    const secureStore = new MemorySecureStore();
    const credentials = new LocalClientCredentialStore(
      join(root, "local-clients.json"),
      secureStore,
    );
    secureStore.set("client-1", "rvc_orphan_token");

    expect(credentials.resolve("client-1")).toBeNull();
    const issued = credentials.issue("client-1");
    expect(credentials.resolve("client-1")).toBe(issued);
    expect(credentials.verify("client-1", issued)).toBe(true);
  });

  it("keeps the previous credential when rotation cannot update the keyring", () => {
    const root = mkdtempSync(join(tmpdir(), "rayvan-credentials-"));
    roots.push(root);
    const secureStore = new MemorySecureStore();
    const credentials = new LocalClientCredentialStore(
      join(root, "local-clients.json"),
      secureStore,
    );
    const original = credentials.issue("client-1");
    secureStore.failNextSet = true;

    expect(() => credentials.rotate("client-1")).toThrow("keyring unavailable");
    expect(credentials.resolve("client-1")).toBe(original);
    expect(credentials.verify("client-1", original)).toBe(true);
  });
});
