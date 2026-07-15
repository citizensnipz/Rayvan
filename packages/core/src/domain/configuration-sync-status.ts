/**
 * Persisted desired/observed/applied sync comparison status.
 *
 * Editor draft dirty is NOT represented here — surface unsaved edits via a
 * separate `editorDirty` / `hasUnsavedLocalChanges` flag on view models.
 * Use `local_changes` only when saved desired differs from last-applied
 * (changes not yet applied to integrations).
 */
export type ConfigurationSyncStatus =
  | "in_sync"
  | "local_changes"
  | "remote_changed"
  | "mismatched"
  | "missing_remote"
  | "missing_local"
  | "partially_applied"
  | "locked"
  | "unknown"
  | "not_managed";
