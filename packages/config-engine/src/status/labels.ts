import type { ConfigurationSyncStatus } from "@rayvan/core";

/**
 * User-facing labels for configuration sync status.
 * Prefer these over colour-only indicators.
 */
export const CONFIGURATION_SYNC_STATUS_LABELS: Record<
  ConfigurationSyncStatus,
  string
> = {
  in_sync: "In sync",
  local_changes: "Changes not applied",
  remote_changed: "Remote changes detected",
  mismatched: "Mismatched",
  missing_remote: "Missing remotely",
  missing_local: "Not managed",
  partially_applied: "Partially in sync",
  locked: "Value locked",
  unknown: "Value unavailable",
  not_managed: "Not managed",
};

export const EDITOR_DIRTY_LABEL = "Unsaved local changes";
export const SYNC_REQUIRED_LABEL = "Sync required";
export const VALUE_UNAVAILABLE_LABEL = "Value unavailable";

export function labelForSyncStatus(
  status: ConfigurationSyncStatus,
  options?: { editorDirty?: boolean; syncRequired?: boolean },
): string {
  if (options?.editorDirty) {
    return EDITOR_DIRTY_LABEL;
  }
  if (options?.syncRequired && status === "in_sync") {
    return SYNC_REQUIRED_LABEL;
  }
  return CONFIGURATION_SYNC_STATUS_LABELS[status];
}
