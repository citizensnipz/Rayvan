import { describe, expect, it } from "vitest";
import {
  configurationKeyId,
  configurationOccurrenceId,
  desiredConfigurationValueId,
  appliedConfigurationStateId,
  environmentId,
  projectId,
  type AppliedConfigurationState,
  type ConfigurationKey,
  type ConfigurationOccurrence,
  type DesiredConfigurationValue,
} from "@rayvan/core";

import { deriveKeyStatus, deriveEnvironmentStatus } from "../src/status/index.js";
import { EDITOR_DIRTY_LABEL } from "../src/status/labels.js";

const PROJECT = projectId("project-1");
const ENV = environmentId("env-1");
const NOW = "2026-07-15T12:00:00.000Z";
const RECENT = "2026-07-14T12:00:00.000Z";
const STALE = "2026-06-01T12:00:00.000Z";

function key(partial: Partial<ConfigurationKey> & Pick<ConfigurationKey, "name">): ConfigurationKey {
  return {
    id: configurationKeyId(partial.id ?? "key-1"),
    projectId: PROJECT,
    name: partial.name,
    valueType: partial.valueType ?? "string",
    required: partial.required ?? false,
    sensitive: partial.sensitive ?? false,
    source: partial.source ?? "manual",
    createdAt: NOW,
    updatedAt: NOW,
    description: partial.description,
  };
}

function desired(
  partial: Partial<DesiredConfigurationValue> = {},
): DesiredConfigurationValue {
  return {
    id: desiredConfigurationValueId(partial.id ?? "desired-1"),
    configurationKeyId: configurationKeyId(partial.configurationKeyId ?? "key-1"),
    environmentId: ENV,
    projectId: PROJECT,
    desiredValue: partial.desiredValue,
    secretValueRef: partial.secretValueRef,
    valueFingerprint: partial.valueFingerprint,
    revision: partial.revision ?? 1,
    createdAt: NOW,
    updatedAt: NOW,
    updatedBy: partial.updatedBy ?? { kind: "user", id: "u1" },
  };
}

function occurrence(
  partial: Partial<ConfigurationOccurrence> = {},
): ConfigurationOccurrence {
  return {
    id: configurationOccurrenceId(partial.id ?? "occ-1"),
    configurationKeyId: configurationKeyId(partial.configurationKeyId ?? "key-1"),
    projectId: PROJECT,
    environmentId: ENV,
    pluginId: partial.pluginId ?? "vercel",
    connectionId: partial.connectionId ?? "conn-1",
    discoveredResourceId: partial.discoveredResourceId ?? "res-1",
    resourceBindingId: partial.resourceBindingId ?? "binding-1",
    providerKey: partial.providerKey ?? "KEY",
    valueAccess: partial.valueAccess ?? "readable",
    observedValue: partial.observedValue,
    maskedValue: partial.maskedValue,
    valueFingerprint: partial.valueFingerprint,
    secretValueRef: partial.secretValueRef,
    firstObservedAt: partial.firstObservedAt ?? RECENT,
    lastObservedAt: partial.lastObservedAt ?? RECENT,
  };
}

function applied(
  partial: Partial<AppliedConfigurationState> = {},
): AppliedConfigurationState {
  return {
    id: appliedConfigurationStateId(partial.id ?? "applied-1"),
    configurationKeyId: configurationKeyId(partial.configurationKeyId ?? "key-1"),
    environmentId: ENV,
    projectId: PROJECT,
    resourceBindingId: partial.resourceBindingId ?? "binding-1",
    desiredRevision: partial.desiredRevision ?? 1,
    appliedFingerprint: partial.appliedFingerprint,
    applyExecutionId: partial.applyExecutionId ?? "exec-1",
    status: partial.status ?? "applied",
    appliedAt: partial.appliedAt ?? RECENT,
    verifiedAt: partial.verifiedAt,
  };
}

describe("deriveKeyStatus", () => {
  it("keeps editor dirty separate from syncStatus (unsaved vs unapplied)", () => {
    const k = key({ name: "API_URL", valueType: "url" });
    const d = desired({
      desiredValue: "https://api.example.com",
      valueFingerprint: "fp:a",
    });
    const status = deriveKeyStatus({
      key: k,
      environmentId: ENV,
      desired: d,
      occurrences: [
        occurrence({
          observedValue: "https://api.example.com",
          valueFingerprint: "fp:a",
        }),
      ],
      applied: [],
      targets: [
        {
          configurationKeyId: k.id,
          environmentId: ENV,
          resourceBindingId: "binding-1",
        },
      ],
      draft: {
        configurationKeyId: k.id,
        draftValue: "https://api.draft.example.com",
        dirty: true,
      },
      now: NOW,
    });

    expect(status.editorDirty).toBe(true);
    expect(status.hasUnsavedLocalChanges).toBe(true);
    expect(status.statusLabel).toBe(EDITOR_DIRTY_LABEL);
    // Persisted: desired matches observed but never applied → local_changes
    expect(status.syncStatus).toBe("local_changes");
  });

  it("reports in_sync when desired, applied, and observed agree", () => {
    const k = key({ name: "NODE_ENV" });
    const d = desired({
      desiredValue: "production",
      valueFingerprint: "fp:prod",
      revision: 2,
    });
    const status = deriveKeyStatus({
      key: k,
      environmentId: ENV,
      desired: d,
      occurrences: [
        occurrence({
          observedValue: "production",
          valueFingerprint: "fp:prod",
        }),
      ],
      applied: [
        applied({
          desiredRevision: 2,
          appliedFingerprint: "fp:prod",
        }),
      ],
      targets: [
        {
          configurationKeyId: k.id,
          environmentId: ENV,
          resourceBindingId: "binding-1",
        },
      ],
      now: NOW,
    });

    expect(status.syncStatus).toBe("in_sync");
    expect(status.editorDirty).toBe(false);
  });

  it("reports remote_changed when observed drifts from last-applied", () => {
    const k = key({ name: "API_URL", valueType: "url" });
    const d = desired({
      desiredValue: "https://api.example.com",
      valueFingerprint: "fp:old",
      revision: 1,
    });
    const status = deriveKeyStatus({
      key: k,
      environmentId: ENV,
      desired: d,
      occurrences: [
        occurrence({
          observedValue: "https://api.hijacked.example.com",
          valueFingerprint: "fp:new",
        }),
      ],
      applied: [
        applied({
          desiredRevision: 1,
          appliedFingerprint: "fp:old",
        }),
      ],
      targets: [
        {
          configurationKeyId: k.id,
          environmentId: ENV,
          resourceBindingId: "binding-1",
        },
      ],
      now: NOW,
    });

    expect(status.syncStatus).toBe("remote_changed");
  });

  it("reports missing_remote when target has no occurrence", () => {
    const k = key({ name: "STRIPE_SECRET_KEY", sensitive: true, valueType: "secret" });
    const d = desired({
      secretValueRef: "cred:stripe",
      valueFingerprint: "fp:stripe",
    });
    const status = deriveKeyStatus({
      key: k,
      environmentId: ENV,
      desired: d,
      occurrences: [],
      applied: [],
      targets: [
        {
          configurationKeyId: k.id,
          environmentId: ENV,
          resourceBindingId: "binding-1",
        },
      ],
      now: NOW,
    });

    expect(status.syncStatus).toBe("missing_remote");
  });

  it("reports missing_local / not_managed when occurrence exists without desired", () => {
    const k = key({ name: "DISCOVERED_ONLY" });
    const status = deriveKeyStatus({
      key: k,
      environmentId: ENV,
      desired: null,
      occurrences: [
        occurrence({
          observedValue: "hello",
          valueFingerprint: "fp:h",
        }),
      ],
      applied: [],
      targets: [
        {
          configurationKeyId: k.id,
          environmentId: ENV,
          resourceBindingId: "binding-1",
        },
      ],
      now: NOW,
    });

    expect(status.syncStatus).toBe("missing_local");
  });

  it("reports partially_applied across resources", () => {
    const k = key({ name: "API_URL", valueType: "url" });
    const d = desired({
      desiredValue: "https://api.example.com",
      valueFingerprint: "fp:a",
      revision: 1,
    });
    const status = deriveKeyStatus({
      key: k,
      environmentId: ENV,
      desired: d,
      occurrences: [
        occurrence({
          id: configurationOccurrenceId("occ-1"),
          resourceBindingId: "binding-1",
          observedValue: "https://api.example.com",
          valueFingerprint: "fp:a",
        }),
        occurrence({
          id: configurationOccurrenceId("occ-2"),
          resourceBindingId: "binding-2",
          discoveredResourceId: "res-2",
          observedValue: "https://other.example.com",
          valueFingerprint: "fp:b",
        }),
      ],
      applied: [
        applied({
          resourceBindingId: "binding-1",
          desiredRevision: 1,
          appliedFingerprint: "fp:a",
        }),
      ],
      targets: [
        {
          configurationKeyId: k.id,
          environmentId: ENV,
          resourceBindingId: "binding-1",
        },
        {
          configurationKeyId: k.id,
          environmentId: ENV,
          resourceBindingId: "binding-2",
        },
      ],
      now: NOW,
    });

    expect(status.syncStatus).toBe("partially_applied");
  });

  it("reports locked and never claims equality", () => {
    const k = key({ name: "GITHUB_TOKEN", sensitive: true, valueType: "secret" });
    const d = desired({
      secretValueRef: "cred:gh",
      valueFingerprint: "fp:gh",
    });
    const status = deriveKeyStatus({
      key: k,
      environmentId: ENV,
      desired: d,
      occurrences: [
        occurrence({
          valueAccess: "locked",
          maskedValue: "••••",
          valueFingerprint: "fp:gh",
        }),
      ],
      applied: [
        applied({
          desiredRevision: 1,
          appliedFingerprint: "fp:gh",
        }),
      ],
      targets: [
        {
          configurationKeyId: k.id,
          environmentId: ENV,
          resourceBindingId: "binding-1",
        },
      ],
      now: NOW,
    });

    expect(status.syncStatus).toBe("locked");
  });

  it("flags stale observed without claiming confident in_sync", () => {
    const k = key({ name: "NODE_ENV" });
    const d = desired({
      desiredValue: "production",
      valueFingerprint: "fp:prod",
      revision: 1,
    });
    const status = deriveKeyStatus({
      key: k,
      environmentId: ENV,
      desired: d,
      occurrences: [
        occurrence({
          observedValue: "production",
          valueFingerprint: "fp:prod",
          lastObservedAt: STALE,
        }),
      ],
      applied: [
        applied({
          desiredRevision: 1,
          appliedFingerprint: "fp:prod",
        }),
      ],
      targets: [
        {
          configurationKeyId: k.id,
          environmentId: ENV,
          resourceBindingId: "binding-1",
        },
      ],
      now: NOW,
    });

    expect(status.syncStatus).toBe("in_sync");
    expect(status.observedMayBeStale).toBe(true);
    expect(status.syncRequired).toBe(true);
    expect(status.statusLabel).toBe("Sync required");
  });

  it("reports unknown when fingerprints are missing for sensitive values", () => {
    const k = key({ name: "SECRET", sensitive: true, valueType: "secret" });
    const d = desired({ secretValueRef: "cred:s" });
    const status = deriveKeyStatus({
      key: k,
      environmentId: ENV,
      desired: d,
      occurrences: [
        occurrence({
          valueAccess: "masked",
          maskedValue: "••••",
        }),
      ],
      applied: [],
      targets: [
        {
          configurationKeyId: k.id,
          environmentId: ENV,
          resourceBindingId: "binding-1",
        },
      ],
      now: NOW,
    });

    expect(status.syncStatus).toBe("unknown");
  });
});

describe("deriveEnvironmentStatus", () => {
  it("aggregates key statuses and headline", () => {
    const keys = [
      key({ id: configurationKeyId("key-1"), name: "A" }),
      key({ id: configurationKeyId("key-2"), name: "B" }),
    ];
    const result = deriveEnvironmentStatus({
      environmentId: ENV,
      keys,
      desired: [
        desired({
          configurationKeyId: configurationKeyId("key-1"),
          desiredValue: "x",
          valueFingerprint: "fp:x",
          revision: 1,
        }),
      ],
      occurrences: [
        occurrence({
          configurationKeyId: configurationKeyId("key-1"),
          observedValue: "x",
          valueFingerprint: "fp:x",
        }),
        occurrence({
          id: configurationOccurrenceId("occ-2"),
          configurationKeyId: configurationKeyId("key-2"),
          observedValue: "y",
          valueFingerprint: "fp:y",
          resourceBindingId: "binding-2",
        }),
      ],
      applied: [
        applied({
          configurationKeyId: configurationKeyId("key-1"),
          desiredRevision: 1,
          appliedFingerprint: "fp:x",
        }),
      ],
      now: NOW,
    });

    expect(result.summary.inSyncCount).toBe(1);
    expect(result.summary.missingLocalCount + result.summary.notManagedCount).toBeGreaterThan(0);
    expect(result.headlineLabel.length).toBeGreaterThan(0);
  });
});
