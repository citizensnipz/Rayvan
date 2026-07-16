import type {
  CredentialReferenceRecord,
  CredentialStorageProvider,
} from "../models.js";

export interface StoreCredentialRequest {
  pluginId: string;
  connectionId: string;
  provider: CredentialStorageProvider;
  credentialType: string;
  displayName?: string;
  secret: unknown;
}

/**
 * Credential material lives outside ordinary DB tables.
 * Repositories store CredentialReferenceRecord only (refs, never secrets).
 */
export interface CredentialStore {
  put(request: StoreCredentialRequest): Promise<CredentialReferenceRecord>;
  get(reference: CredentialReferenceRecord): Promise<unknown>;
  delete(reference: CredentialReferenceRecord): Promise<void>;
  exists(reference: CredentialReferenceRecord): Promise<boolean>;
}
