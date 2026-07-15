import type { ConfigurationSyncStatus } from "@rayvan/core";

import {
  appliedMatchesOccurrence,
  desiredMatchesApplied,
  desiredMatchesOccurrence,
  isLockedAccess,
  isObservedStale,
  DEFAULT_STALE_AFTER_MS,
} from "./compare.js";
import { labelForSyncStatus } from "./labels.js";
import type {
  ConfigurationKeyStatusViewModel,
  ConfigurationResourceStatusSummary,
  ConfigurationTargetRef,
  DeriveEnvironmentStatusInput,
  DeriveKeyStatusInput,
  DeriveResourceStatusInput,
  EnvironmentConfigurationStatusViewModel,
  EnvironmentConfigurationStatusService,
  ResourceConfigurationStatusViewModel,
} from "./types.js";

/**
 * Status precedence (persisted comparison only — editor dirty is separate):
 *
 * 1. not_managed / missing_local — occurrence exists, no desired
 * 2. missing_remote — target expects key but occurrence missing / valueAccess missing
 * 3. locked — cannot compare (locked / name_only / masked without fingerprint)
 * 4. remote_changed — observed differs from last-applied while desired still matches last-applied
 * 5. mismatched — saved desired differs from readable observed
 * 6. local_changes — desired differs from last-applied (changes not applied)
 * 7. partially_applied — some resources match, others don't
 * 8. in_sync — desired matches observed (and applied when present)
 * 9. unknown — insufficient information
 *
 * Editor draft dirty → `editorDirty` / `hasUnsavedLocalChanges` only.
 * Prefer UI label "Unsaved local changes"; syncStatus still from persisted state.
 */

function deriveTargetsFromOccurrences(
  environmentId: string,
  occurrences: DeriveKeyStatusInput["occurrences"],
  configurationKeyId: string,
): ConfigurationTargetRef[] {
  const targets: ConfigurationTargetRef[] = [];
  const seen = new Set<string>();
  for (const occurrence of occurrences) {
    if (
      occurrence.configurationKeyId !== configurationKeyId ||
      occurrence.environmentId !== environmentId ||
      !occurrence.resourceBindingId
    ) {
      continue;
    }
    if (seen.has(occurrence.resourceBindingId)) {
      continue;
    }
    seen.add(occurrence.resourceBindingId);
    targets.push({
      configurationKeyId,
      environmentId,
      resourceBindingId: occurrence.resourceBindingId,
      pluginId: occurrence.pluginId,
      discoveredResourceId: occurrence.discoveredResourceId,
    });
  }
  return targets;
}

function emptySummary(): EnvironmentConfigurationStatusViewModel["summary"] {
  return {
    inSyncCount: 0,
    localChangesCount: 0,
    remoteChangedCount: 0,
    mismatchedCount: 0,
    missingRemoteCount: 0,
    missingLocalCount: 0,
    notManagedCount: 0,
    partiallyAppliedCount: 0,
    lockedCount: 0,
    unknownCount: 0,
    unsavedDraftCount: 0,
    staleObservedCount: 0,
  };
}

function bumpSummary(
  summary: EnvironmentConfigurationStatusViewModel["summary"],
  status: ConfigurationSyncStatus,
): void {
  switch (status) {
    case "in_sync":
      summary.inSyncCount += 1;
      break;
    case "local_changes":
      summary.localChangesCount += 1;
      break;
    case "remote_changed":
      summary.remoteChangedCount += 1;
      break;
    case "mismatched":
      summary.mismatchedCount += 1;
      break;
    case "missing_remote":
      summary.missingRemoteCount += 1;
      break;
    case "missing_local":
      summary.missingLocalCount += 1;
      // Same user-facing meaning as not_managed in v1 (no targets table).
      summary.notManagedCount += 1;
      break;
    case "not_managed":
      summary.notManagedCount += 1;
      summary.missingLocalCount += 1;
      break;
    case "partially_applied":
      summary.partiallyAppliedCount += 1;
      break;
    case "locked":
      summary.lockedCount += 1;
      break;
    case "unknown":
      summary.unknownCount += 1;
      break;
  }
}

function headlineFromSummary(
  summary: EnvironmentConfigurationStatusViewModel["summary"],
): string {
  if (summary.unsavedDraftCount > 0) {
    return "Unsaved local changes";
  }
  if (summary.localChangesCount > 0) {
    return "Changes not applied";
  }
  if (summary.remoteChangedCount > 0) {
    return "Remote changes detected";
  }
  if (summary.mismatchedCount > 0 || summary.partiallyAppliedCount > 0) {
    return "Partially in sync";
  }
  if (summary.missingRemoteCount > 0) {
    return "Missing remotely";
  }
  if (summary.lockedCount > 0) {
    return "Value locked";
  }
  if (summary.staleObservedCount > 0 && summary.inSyncCount > 0) {
    return "Sync required";
  }
  if (
    summary.inSyncCount > 0 &&
    summary.notManagedCount === 0 &&
    summary.missingLocalCount === 0
  ) {
    return "In sync";
  }
  if (summary.notManagedCount > 0 || summary.missingLocalCount > 0) {
    return "Not managed";
  }
  return "Value unavailable";
}

export function deriveKeyStatus(
  input: DeriveKeyStatusInput,
): ConfigurationKeyStatusViewModel {
  const now = input.now ?? new Date().toISOString();
  const staleAfterMs = input.staleAfterMs ?? DEFAULT_STALE_AFTER_MS;
  const envOccurrences = input.occurrences.filter(
    (occurrence) =>
      occurrence.configurationKeyId === input.key.id &&
      occurrence.environmentId === input.environmentId,
  );
  const targets =
    input.targets.length > 0
      ? input.targets
      : deriveTargetsFromOccurrences(
          input.environmentId,
          envOccurrences,
          input.key.id,
        );
  const appliedForKey = input.applied.filter(
    (state) =>
      state.configurationKeyId === input.key.id &&
      state.environmentId === input.environmentId,
  );
  const desired = input.desired ?? null;
  const editorDirty = Boolean(input.draft?.dirty);
  const observedMayBeStale = isObservedStale(
    envOccurrences,
    now,
    staleAfterMs,
  );

  const resourceStatuses: ConfigurationResourceStatusSummary[] = [];

  for (const target of targets) {
    const occurrence = envOccurrences.find(
      (item) => item.resourceBindingId === target.resourceBindingId,
    );
    const applied = appliedForKey.find(
      (state) => state.resourceBindingId === target.resourceBindingId,
    );
    let resourceStatus: ConfigurationSyncStatus = "unknown";

    if (!desired) {
      resourceStatus =
        occurrence && occurrence.valueAccess !== "missing"
          ? "not_managed"
          : "missing_remote";
    } else if (!occurrence || occurrence.valueAccess === "missing") {
      resourceStatus = "missing_remote";
    } else if (isLockedAccess(occurrence.valueAccess)) {
      resourceStatus = "locked";
    } else {
      const desiredVsObserved = desiredMatchesOccurrence(
        input.key,
        desired,
        occurrence,
      );
      const appliedOk = desiredMatchesApplied(desired, applied);
      const appliedVsObserved = applied
        ? appliedMatchesOccurrence(input.key, applied, occurrence)
        : undefined;

      if (
        applied &&
        appliedOk &&
        appliedVsObserved === "mismatch" &&
        desiredVsObserved === "mismatch"
      ) {
        // Observed drifted from last-applied while desired still equals applied.
        resourceStatus = "remote_changed";
      } else if (desiredVsObserved === "mismatch") {
        // Desired differs from observed. If desired also differs from last-applied,
        // surface as changes not applied; otherwise mismatched drift.
        resourceStatus = appliedOk ? "mismatched" : "local_changes";
      } else if (desiredVsObserved === "locked") {
        resourceStatus = "locked";
      } else if (desiredVsObserved === "missing") {
        resourceStatus = "missing_remote";
      } else if (desiredVsObserved === "match") {
        resourceStatus = appliedOk ? "in_sync" : "local_changes";
      } else if (!appliedOk && applied) {
        resourceStatus = "local_changes";
      } else if (!applied) {
        resourceStatus =
          desiredVsObserved === "unknown" ? "unknown" : "local_changes";
      } else {
        resourceStatus = "unknown";
      }
    }

    resourceStatuses.push({
      resourceBindingId: target.resourceBindingId,
      pluginId: target.pluginId ?? occurrence?.pluginId,
      syncStatus: resourceStatus,
      statusLabel: labelForSyncStatus(resourceStatus),
      occurrenceId: occurrence?.id,
      appliedStatus: applied?.status,
    });
  }

  // Occurrences without binding still inform not_managed / missing_local.
  const unboundOccurrences = envOccurrences.filter(
    (occurrence) => !occurrence.resourceBindingId,
  );

  let syncStatus: ConfigurationSyncStatus;

  if (!desired) {
    if (envOccurrences.some((occurrence) => occurrence.valueAccess !== "missing")) {
      syncStatus = "not_managed";
    } else if (targets.length > 0) {
      syncStatus = "missing_remote";
    } else {
      syncStatus = "not_managed";
    }
  } else if (targets.length === 0 && unboundOccurrences.length === 0) {
    // Desired saved but no targets/occurrences in this environment.
    syncStatus = "missing_remote";
  } else if (resourceStatuses.length === 0) {
    const first = unboundOccurrences[0];
    if (!first) {
      syncStatus = "missing_remote";
    } else if (isLockedAccess(first.valueAccess)) {
      syncStatus = "locked";
    } else {
      const result = desiredMatchesOccurrence(input.key, desired, first);
      if (result === "match") {
        syncStatus =
          appliedForKey.length === 0 ||
          appliedForKey.every((state) => desiredMatchesApplied(desired, state))
            ? appliedForKey.length === 0
              ? "local_changes"
              : "in_sync"
            : "local_changes";
      } else if (result === "mismatch") {
        syncStatus = "mismatched";
      } else if (result === "locked") {
        syncStatus = "locked";
      } else if (result === "missing") {
        syncStatus = "missing_remote";
      } else {
        syncStatus = "unknown";
      }
    }
  } else {
    const statuses = new Set(resourceStatuses.map((item) => item.syncStatus));
    if (statuses.size === 1) {
      syncStatus = resourceStatuses[0]!.syncStatus;
    } else if (
      statuses.has("in_sync") &&
      (statuses.has("mismatched") ||
        statuses.has("local_changes") ||
        statuses.has("missing_remote") ||
        statuses.has("remote_changed") ||
        statuses.has("locked"))
    ) {
      syncStatus = "partially_applied";
    } else if (statuses.has("remote_changed")) {
      syncStatus = "remote_changed";
    } else if (statuses.has("mismatched")) {
      syncStatus = "mismatched";
    } else if (statuses.has("local_changes")) {
      syncStatus = "local_changes";
    } else if (statuses.has("missing_remote")) {
      syncStatus = "missing_remote";
    } else if (statuses.has("locked")) {
      syncStatus = "locked";
    } else {
      syncStatus = "unknown";
    }
  }

  // missing_local alias when occurrence exists without desired
  if (!desired && envOccurrences.length > 0) {
    syncStatus = "missing_local";
  }

  const syncRequired =
    observedMayBeStale &&
    (syncStatus === "in_sync" || syncStatus === "unknown");

  return {
    configurationKeyId: input.key.id,
    configurationKeyName: input.key.name,
    environmentId: input.environmentId,
    syncStatus,
    statusLabel: labelForSyncStatus(syncStatus, {
      editorDirty,
      syncRequired,
    }),
    editorDirty,
    hasUnsavedLocalChanges: editorDirty,
    observedMayBeStale,
    syncRequired,
    desiredRevision: desired?.revision,
    resourceStatuses,
  };
}

export function deriveEnvironmentStatus(
  input: DeriveEnvironmentStatusInput,
): EnvironmentConfigurationStatusViewModel {
  const draftByKey = new Map(
    (input.drafts ?? []).map((draft) => [draft.configurationKeyId, draft]),
  );
  const desiredByKey = new Map(
    input.desired.map((value) => [value.configurationKeyId, value]),
  );
  const keyStatuses = input.keys.map((key) => {
    const envOccurrences = input.occurrences.filter(
      (occurrence) =>
        occurrence.configurationKeyId === key.id &&
        occurrence.environmentId === input.environmentId,
    );
    return deriveKeyStatus({
      key,
      environmentId: input.environmentId,
      desired: desiredByKey.get(key.id),
      occurrences: envOccurrences,
      applied: input.applied.filter(
        (state) => state.configurationKeyId === key.id,
      ),
      targets: deriveTargetsFromOccurrences(
        input.environmentId,
        envOccurrences,
        key.id,
      ),
      draft: draftByKey.get(key.id),
      now: input.now,
      staleAfterMs: input.staleAfterMs,
    });
  });

  // Include keys that only appear via occurrences (discovered not in keys list)
  // — callers should pass the full key set; we still filter to relevant keys.
  const relevant = keyStatuses.filter((status) => {
    const hasDesired = desiredByKey.has(status.configurationKeyId);
    const hasOccurrence = input.occurrences.some(
      (occurrence) =>
        occurrence.configurationKeyId === status.configurationKeyId &&
        occurrence.environmentId === input.environmentId,
    );
    return hasDesired || hasOccurrence;
  });

  const summary = emptySummary();
  for (const status of relevant) {
    bumpSummary(summary, status.syncStatus);
    if (status.hasUnsavedLocalChanges) {
      summary.unsavedDraftCount += 1;
    }
    if (status.observedMayBeStale) {
      summary.staleObservedCount += 1;
    }
  }

  return {
    environmentId: input.environmentId,
    keyStatuses: relevant,
    summary,
    headlineLabel: headlineFromSummary(summary),
    hasUnsavedLocalChanges: summary.unsavedDraftCount > 0,
    hasChangesNotApplied: summary.localChangesCount > 0,
  };
}

export function deriveResourceStatus(
  input: DeriveResourceStatusInput,
): ResourceConfigurationStatusViewModel {
  const envStatus = deriveEnvironmentStatus({
    environmentId: input.environmentId,
    keys: input.keys,
    desired: input.desired,
    occurrences: input.occurrences.filter(
      (occurrence) =>
        occurrence.resourceBindingId === input.resourceBindingId ||
        (!occurrence.resourceBindingId &&
          occurrence.environmentId === input.environmentId),
    ),
    applied: input.applied.filter(
      (state) => state.resourceBindingId === input.resourceBindingId,
    ),
    now: input.now,
    staleAfterMs: input.staleAfterMs,
  });

  const keyStatuses = envStatus.keyStatuses
    .map((status) => ({
      ...status,
      resourceStatuses: status.resourceStatuses.filter(
        (resource) => resource.resourceBindingId === input.resourceBindingId,
      ),
    }))
    .filter(
      (status) =>
        status.resourceStatuses.length > 0 ||
        status.syncStatus === "not_managed" ||
        status.syncStatus === "missing_local",
    );

  const summary = emptySummary();
  for (const status of keyStatuses) {
    bumpSummary(summary, status.syncStatus);
    if (status.observedMayBeStale) {
      summary.staleObservedCount += 1;
    }
  }

  return {
    environmentId: input.environmentId,
    resourceBindingId: input.resourceBindingId,
    keyStatuses,
    summary,
    headlineLabel: headlineFromSummary(summary),
  };
}

export class DefaultEnvironmentConfigurationStatusService
  implements EnvironmentConfigurationStatusService
{
  getEnvironmentStatus(
    input: DeriveEnvironmentStatusInput,
  ): EnvironmentConfigurationStatusViewModel {
    return deriveEnvironmentStatus(input);
  }

  getKeyStatus(input: DeriveKeyStatusInput): ConfigurationKeyStatusViewModel {
    return deriveKeyStatus(input);
  }

  getResourceStatus(
    input: DeriveResourceStatusInput,
  ): ResourceConfigurationStatusViewModel {
    return deriveResourceStatus(input);
  }
}
