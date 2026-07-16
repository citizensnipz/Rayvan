import type { CredentialReferenceRecord } from "../models.js";
import type { CredentialStore, StoreCredentialRequest } from "./types.js";

/**
 * In-memory credential store for tests and local development.
 * Not a production secrets backend — do not use as encrypted_local_store.
 */
export class DevelopmentMemoryCredentialStore implements CredentialStore {
  private readonly secrets = new Map<string, unknown>();
  private readonly references = new Map<string, CredentialReferenceRecord>();

  async put(request: StoreCredentialRequest): Promise<CredentialReferenceRecord> {
    const now = new Date().toISOString();
    const id = crypto.randomUUID();
    const storageKey = `dev:${request.connectionId}:${id}`;
    const record: CredentialReferenceRecord = {
      id,
      pluginId: request.pluginId,
      connectionId: request.connectionId,
      provider: "development_memory",
      storageKey,
      credentialType: request.credentialType,
      displayName: request.displayName,
      createdAt: now,
      updatedAt: now,
    };
    this.secrets.set(storageKey, request.secret);
    this.references.set(id, record);
    return record;
  }

  async get(reference: CredentialReferenceRecord): Promise<unknown> {
    if (!this.secrets.has(reference.storageKey)) {
      throw new Error(`Credential not found: ${reference.id}`);
    }
    return this.secrets.get(reference.storageKey);
  }

  async delete(reference: CredentialReferenceRecord): Promise<void> {
    this.secrets.delete(reference.storageKey);
    this.references.delete(reference.id);
  }

  async exists(reference: CredentialReferenceRecord): Promise<boolean> {
    return this.secrets.has(reference.storageKey);
  }
}
