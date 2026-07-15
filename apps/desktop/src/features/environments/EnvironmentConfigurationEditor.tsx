import { useEffect, useMemo, useState, type CSSProperties } from "react";
import type {
  AppliedConfigurationState,
  ConfigurationKey,
  ConfigurationOccurrence,
  DesiredConfigurationValue,
  Environment,
} from "@rayvan/core";
import type {
  ConfigurationApplyPlan,
  ConfigurationApplyResult,
  EnvironmentConfigurationStatusViewModel,
} from "@rayvan/config-engine";
import { Button } from "@rayvan/ui";

import type { EnvironmentsGateway } from "../../lib/environments/index.js";
import {
  fingerprintForDraft,
  inputTypeForValueType,
  type EnvironmentConfigurationDraftValue,
} from "./view-models.js";
import { ApplyConfigurationReview } from "./ApplyConfigurationReview.js";

const rowStyle: CSSProperties = {
  border: "1px solid var(--color-border)",
  borderRadius: "8px",
  background: "var(--color-surface)",
  padding: "0.75rem",
};

interface EnvironmentConfigurationEditorProps {
  projectId: string;
  environment: Environment;
  keys: ConfigurationKey[];
  occurrences: ConfigurationOccurrence[];
  desired: DesiredConfigurationValue[];
  applied: AppliedConfigurationState[];
  status: EnvironmentConfigurationStatusViewModel | null;
  gateway: EnvironmentsGateway;
  onOpenMatrix: () => void;
  onRefresh: () => Promise<void>;
  onBanner: (message: string) => void;
}

function draftFromDesired(
  key: ConfigurationKey,
  desired: DesiredConfigurationValue | undefined,
): EnvironmentConfigurationDraftValue {
  if (key.sensitive || key.valueType === "secret") {
    return {
      configurationKeyId: key.id,
      draftValue: "",
      draftSecretValueRef: desired?.secretValueRef,
      draftFingerprint: desired?.valueFingerprint,
      dirty: false,
    };
  }
  return {
    configurationKeyId: key.id,
    draftValue: desired?.desiredValue ?? "",
    draftFingerprint: desired?.valueFingerprint,
    dirty: false,
  };
}

export function EnvironmentConfigurationEditor({
  projectId,
  environment,
  keys,
  occurrences,
  desired,
  applied,
  status,
  gateway,
  onOpenMatrix,
  onRefresh,
  onBanner,
}: EnvironmentConfigurationEditorProps) {
  const envOccurrences = useMemo(
    () =>
      occurrences.filter(
        (occurrence) => occurrence.environmentId === environment.id,
      ),
    [occurrences, environment.id],
  );
  const desiredByKey = useMemo(
    () =>
      new Map(
        desired.map((value) => [String(value.configurationKeyId), value]),
      ),
    [desired],
  );
  const statusByKey = useMemo(
    () =>
      new Map(
        (status?.keyStatuses ?? []).map((item) => [
          String(item.configurationKeyId),
          item,
        ]),
      ),
    [status],
  );

  const relevantKeys = useMemo(() => {
    const keyIds = new Set<string>();
    for (const value of desired) {
      keyIds.add(String(value.configurationKeyId));
    }
    for (const occurrence of envOccurrences) {
      keyIds.add(String(occurrence.configurationKeyId));
    }
    return keys
      .filter((key) => keyIds.has(String(key.id)))
      .sort((left, right) => left.name.localeCompare(right.name));
  }, [keys, desired, envOccurrences]);

  const [drafts, setDrafts] = useState<
    Record<string, EnvironmentConfigurationDraftValue>
  >({});
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [revealed, setRevealed] = useState<Record<string, boolean>>({});
  const [saving, setSaving] = useState(false);
  const [conflictMessage, setConflictMessage] = useState<string | null>(null);
  const [applyPlan, setApplyPlan] = useState<ConfigurationApplyPlan | null>(null);
  const [applyResult, setApplyResult] = useState<ConfigurationApplyResult | null>(
    null,
  );

  useEffect(() => {
    const next: Record<string, EnvironmentConfigurationDraftValue> = {};
    for (const key of relevantKeys) {
      next[key.id] = draftFromDesired(key, desiredByKey.get(key.id));
    }
    setDrafts(next);
    setRevealed({});
    setConflictMessage(null);
  }, [relevantKeys, desiredByKey]);

  useEffect(() => {
    return () => {
      setRevealed({});
    };
  }, [environment.id]);

  const dirtyDrafts = Object.values(drafts).filter((draft) => draft.dirty);
  const hasUnsaved = dirtyDrafts.length > 0;
  const headerLabel = hasUnsaved
    ? "Unsaved local changes"
    : status?.hasChangesNotApplied
      ? "Changes not applied"
      : (status?.headlineLabel ?? "Configuration");

  function updateDraft(keyId: string, value: string, key: ConfigurationKey) {
    setDrafts((current) => {
      const existing = current[keyId] ?? draftFromDesired(key, desiredByKey.get(keyId));
      const saved = desiredByKey.get(keyId);
      const sensitive = key.sensitive || key.valueType === "secret";
      if (sensitive) {
        // Empty draft means "keep saved secret" (password fields start empty).
        // Only mark dirty when the user types a new plaintext value.
        if (value.length === 0) {
          return {
            ...current,
            [keyId]: {
              ...existing,
              draftValue: "",
              draftSecretValueRef: saved?.secretValueRef ?? existing.draftSecretValueRef,
              draftFingerprint:
                saved?.valueFingerprint ?? existing.draftFingerprint,
              dirty: false,
            },
          };
        }
        // Ephemeral plaintext for editing only — persist as secretValueRef + fingerprint on save.
        const fingerprint = fingerprintForDraft(value);
        const dirty = !saved || saved.valueFingerprint !== fingerprint;
        return {
          ...current,
          [keyId]: {
            ...existing,
            draftValue: value,
            draftSecretValueRef: `cred:local-${key.name.toLowerCase()}`,
            draftFingerprint: fingerprint,
            dirty,
          },
        };
      }
      const dirty = (saved?.desiredValue ?? "") !== value;
      return {
        ...current,
        [keyId]: {
          ...existing,
          draftValue: value,
          draftFingerprint: value ? fingerprintForDraft(value) : undefined,
          dirty,
        },
      };
    });
  }

  async function handleSave() {
    setSaving(true);
    setConflictMessage(null);
    try {
      const inputs = dirtyDrafts.map((draft) => {
        const key = relevantKeys.find((item) => item.id === draft.configurationKeyId)!;
        const saved = desiredByKey.get(draft.configurationKeyId);
        const sensitive = key.sensitive || key.valueType === "secret";
        return {
          configurationKeyId: draft.configurationKeyId,
          environmentId: environment.id,
          projectId,
          desiredValue: sensitive ? undefined : draft.draftValue,
          secretValueRef: sensitive ? draft.draftSecretValueRef : undefined,
          valueFingerprint: draft.draftFingerprint,
          expectedRevision: saved?.revision,
        };
      });
      await gateway.saveDesiredValues(inputs);
      onBanner("Configuration saved locally.");
      await onRefresh();
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to save configuration.";
      if (/revision mismatch/i.test(message)) {
        setConflictMessage(
          "Someone else saved this configuration. Reload and try again.",
        );
      } else {
        setConflictMessage(message);
      }
    } finally {
      setSaving(false);
    }
  }

  function handleDiscard() {
    const next: Record<string, EnvironmentConfigurationDraftValue> = {};
    for (const key of relevantKeys) {
      next[key.id] = draftFromDesired(key, desiredByKey.get(key.id));
    }
    setDrafts(next);
    setRevealed({});
    setConflictMessage(null);
  }

  async function handleApplyClick() {
    if (hasUnsaved) {
      onBanner("Save locally before applying to integrations.");
      return;
    }
    const plan = await gateway.buildApplyPlan(projectId, environment.id);
    setApplyPlan(plan);
    setApplyResult(null);
  }

  async function handleApproveApply() {
    if (!applyPlan) {
      return;
    }
    const result = await gateway.approveApplyPlan(applyPlan.id);
    setApplyResult(result);
    onBanner(
      result.status === "completed"
        ? "Apply completed (stub — no provider calls)."
        : result.status === "partial"
          ? "Apply finished with partial failures. Desired values were preserved."
          : "Apply failed (stub).",
    );
    await onRefresh();
  }

  async function handleAdopt(keyId: string) {
    await gateway.adoptDiscoveredKey({
      configurationKeyId: keyId,
      environmentId: environment.id,
      projectId,
      copyReadableValue: true,
    });
    onBanner("Key added to local configuration.");
    await onRefresh();
  }

  return (
    <div style={{ display: "grid", gap: "0.75rem" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          gap: "0.75rem",
          flexWrap: "wrap",
          alignItems: "center",
        }}
      >
        <div>
          <strong aria-live="polite">{headerLabel}</strong>
          <div style={{ fontSize: "0.85rem", color: "var(--color-text-secondary)" }}>
            {status
              ? `${status.summary.inSyncCount} in sync · ${status.summary.localChangesCount} not applied · ${status.summary.missingRemoteCount} missing remotely`
              : "Loading status…"}
          </div>
        </div>
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          <Button onClick={onOpenMatrix}>Compare with other environments</Button>
          <Button onClick={() => void handleSave()} disabled={!hasUnsaved || saving}>
            Save locally
          </Button>
          <Button onClick={handleDiscard} disabled={!hasUnsaved}>
            Discard
          </Button>
          <Button
            onClick={() => void handleApplyClick()}
            disabled={hasUnsaved}
            title={
              hasUnsaved
                ? "Save locally before applying"
                : "Build an apply plan (no provider calls)"
            }
          >
            Apply to integrations
          </Button>
        </div>
      </div>

      {conflictMessage ? (
        <div role="alert" style={{ color: "var(--color-danger, #b91c1c)" }}>
          {conflictMessage}
        </div>
      ) : null}

      {relevantKeys.length === 0 ? (
        <p style={{ color: "var(--color-text-secondary)" }}>
          No configuration keys for this environment yet.
        </p>
      ) : (
        <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: "0.5rem" }}>
          {relevantKeys.map((key) => {
            const draft = drafts[key.id] ?? draftFromDesired(key, desiredByKey.get(key.id));
            const keyStatus = statusByKey.get(key.id);
            const keyOccurrences = envOccurrences.filter(
              (occurrence) => occurrence.configurationKeyId === key.id,
            );
            const keyApplied = applied.filter(
              (state) => state.configurationKeyId === key.id,
            );
            const sensitive = key.sensitive || key.valueType === "secret";
            const isExpanded = Boolean(expanded[key.id]);
            const isRevealed = Boolean(revealed[key.id]);
            const lockedRemote = keyOccurrences.some(
              (occurrence) =>
                occurrence.valueAccess === "locked" ||
                occurrence.valueAccess === "name_only",
            );
            const notManaged =
              keyStatus?.syncStatus === "not_managed" ||
              keyStatus?.syncStatus === "missing_local";

            return (
              <li key={key.id} style={rowStyle}>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    gap: "0.75rem",
                    flexWrap: "wrap",
                    alignItems: "flex-start",
                  }}
                >
                  <div style={{ minWidth: 0, flex: "1 1 12rem" }}>
                    <strong>{key.name}</strong>
                    <div
                      style={{
                        fontSize: "0.8rem",
                        color: "var(--color-text-secondary)",
                      }}
                    >
                      {key.valueType}
                      {sensitive ? " · sensitive" : ""}
                      {" · "}
                      {keyStatus?.statusLabel ?? "Unknown"}
                      {draft.dirty ? " · Unsaved local changes" : ""}
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: "0.35rem", flexWrap: "wrap" }}>
                    {notManaged ? (
                      <Button onClick={() => void handleAdopt(key.id)}>
                        Add to local configuration
                      </Button>
                    ) : null}
                    <Button
                      onClick={() =>
                        setExpanded((current) => ({
                          ...current,
                          [key.id]: !current[key.id],
                        }))
                      }
                    >
                      {isExpanded ? "Hide resources" : "Show resources"}
                    </Button>
                  </div>
                </div>

                {!notManaged ? (
                  <div style={{ marginTop: "0.65rem" }}>
                    <label
                      htmlFor={`desired-${key.id}`}
                      style={{
                        display: "block",
                        fontSize: "0.8rem",
                        marginBottom: "0.25rem",
                      }}
                    >
                      Desired value
                    </label>
                    {key.valueType === "boolean" ? (
                      <input
                        id={`desired-${key.id}`}
                        type="checkbox"
                        checked={draft.draftValue === "true"}
                        onChange={(event) =>
                          updateDraft(
                            key.id,
                            event.target.checked ? "true" : "false",
                            key,
                          )
                        }
                        aria-label={`Desired value for ${key.name}`}
                      />
                    ) : (
                      <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                        <input
                          id={`desired-${key.id}`}
                          type={inputTypeForValueType(key.valueType, isRevealed, {
                            sensitive,
                          })}
                          value={
                            sensitive && !isRevealed && !draft.dirty
                              ? ""
                              : draft.draftValue
                          }
                          placeholder={
                            sensitive && !isRevealed && !draft.dirty && draft.draftSecretValueRef
                              ? "••••••••"
                              : undefined
                          }
                          onChange={(event) =>
                            updateDraft(key.id, event.target.value, key)
                          }
                          aria-label={`Desired value for ${key.name}`}
                          autoComplete="off"
                          style={{
                            flex: 1,
                            padding: "0.4rem 0.55rem",
                            borderRadius: "6px",
                            border: "1px solid var(--color-border)",
                            background: "var(--color-surface)",
                            color: "var(--color-text)",
                          }}
                        />
                        {sensitive ? (
                          <Button
                            aria-label={
                              isRevealed
                                ? `Hide value for ${key.name}`
                                : `Show value for ${key.name}`
                            }
                            onClick={() =>
                              setRevealed((current) => ({
                                ...current,
                                [key.id]: !current[key.id],
                              }))
                            }
                            disabled={lockedRemote && !draft.dirty}
                            title={
                              lockedRemote && !draft.dirty
                                ? "Remote value is locked and cannot be revealed"
                                : undefined
                            }
                          >
                            {isRevealed ? "Hide" : "Show"}
                          </Button>
                        ) : null}
                      </div>
                    )}
                    {sensitive ? (
                      <div
                        style={{
                          fontSize: "0.75rem",
                          color: "var(--color-text-muted)",
                          marginTop: "0.25rem",
                        }}
                      >
                        Saved as secret reference
                        {draft.draftSecretValueRef
                          ? ` (${draft.draftSecretValueRef})`
                          : ""}
                        — plaintext is never stored in ordinary tables.
                      </div>
                    ) : null}
                  </div>
                ) : null}

                {isExpanded ? (
                  <ul
                    style={{
                      listStyle: "none",
                      padding: "0.65rem 0 0",
                      margin: "0.65rem 0 0",
                      borderTop: "1px solid var(--color-border)",
                      display: "grid",
                      gap: "0.35rem",
                    }}
                  >
                    {keyOccurrences.length === 0 ? (
                      <li style={{ color: "var(--color-text-secondary)", fontSize: "0.85rem" }}>
                        No observed resources for this key.
                      </li>
                    ) : (
                      keyOccurrences.map((occurrence) => {
                        const appliedState = keyApplied.find(
                          (state) =>
                            state.resourceBindingId === occurrence.resourceBindingId,
                        );
                        const resourceStatus = keyStatus?.resourceStatuses.find(
                          (item) =>
                            item.resourceBindingId === occurrence.resourceBindingId,
                        );
                        const locked =
                          occurrence.valueAccess === "locked" ||
                          occurrence.valueAccess === "name_only";
                        return (
                          <li
                            key={occurrence.id}
                            style={{ fontSize: "0.85rem", color: "var(--color-text-secondary)" }}
                          >
                            <strong style={{ color: "var(--color-text)" }}>
                              {occurrence.pluginId}
                            </strong>
                            {" · "}
                            {resourceStatus?.statusLabel ?? occurrence.valueAccess}
                            {" · observed: "}
                            {locked
                              ? "locked"
                              : sensitive
                                ? (occurrence.maskedValue ?? "••••")
                                : (occurrence.observedValue ?? "—")}
                            {appliedState
                              ? ` · applied: ${appliedState.status}`
                              : " · not applied"}
                          </li>
                        );
                      })
                    )}
                  </ul>
                ) : null}
              </li>
            );
          })}
        </ul>
      )}

      {applyPlan ? (
        <ApplyConfigurationReview
          plan={applyPlan}
          result={applyResult}
          onApprove={() => void handleApproveApply()}
          onClose={() => {
            setApplyPlan(null);
            setApplyResult(null);
          }}
        />
      ) : null}
    </div>
  );
}
