import { existsSync, mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { SecureCredentialStore } from "@rayvan/daemon-client";
import { afterEach, describe, expect, it } from "vitest";

import { DaemonSecretStore } from "../src/services/secrets.js";

class MemorySecureStore implements SecureCredentialStore {
  readonly values = new Map<string, string>();

  set(account: string, value: string): void {
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

describe("DaemonSecretStore", () => {
  it("stores values in secure storage without plaintext blob files", () => {
    const root = mkdtempSync(join(tmpdir(), "rayvan-secrets-"));
    roots.push(root);
    const secureStore = new MemorySecureStore();
    const secrets = new DaemonSecretStore(root, secureStore);

    const saved = secrets.put("provider-secret");

    expect(secrets.get(saved.ref)).toBe("provider-secret");
    expect(saved.fingerprint).not.toContain("provider-secret");
    expect(existsSync(join(root, `${saved.ref}.bin`))).toBe(false);
    expect(readdirSync(root)).toEqual([]);
  });

  it("deletes values from secure storage", () => {
    const root = mkdtempSync(join(tmpdir(), "rayvan-secrets-"));
    roots.push(root);
    const secureStore = new MemorySecureStore();
    const secrets = new DaemonSecretStore(root, secureStore);
    const saved = secrets.put("provider-secret");

    secrets.delete(saved.ref);

    expect(secrets.get(saved.ref)).toBeNull();
  });
});
