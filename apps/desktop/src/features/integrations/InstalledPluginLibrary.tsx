import { useState, type CSSProperties, type FormEvent } from "react";
import { PLUGIN_PERMISSIONS, type PluginPermission } from "@rayvan/plugin-sdk";
import { Button, Input } from "@rayvan/ui";

import { IntegrationIcon } from "./icons.js";
import { resolveIntegrationTheme } from "./theme.js";
import type { LibraryPluginViewModel } from "./view-models.js";

const listStyle: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: "0.75rem",
  maxHeight: "22rem",
  overflowY: "auto",
};

const itemStyle: CSSProperties = {
  display: "flex",
  gap: "0.75rem",
  alignItems: "flex-start",
  padding: "0.75rem",
  borderRadius: "8px",
  border: "1px solid var(--color-border)",
};

const badgeStyle: CSSProperties = {
  display: "inline-block",
  fontSize: "0.7rem",
  fontWeight: 600,
  padding: "0.1rem 0.45rem",
  borderRadius: "999px",
  background: "var(--color-surface-muted)",
  color: "var(--color-text-secondary)",
  marginLeft: "0.4rem",
};

const PERMISSION_LABELS: Record<PluginPermission, string> = {
  network: "Access the network",
  read_secrets: "Read stored secrets",
  write_remote_configuration: "Write remote configuration",
  read_local_files: "Read local files",
  write_local_files: "Write local files",
};

const DEFAULT_CHECKED_PERMISSIONS: readonly PluginPermission[] = ["network"];

export interface AddIntegrationSubmission {
  installedPluginId: string;
  connectionName: string;
  permissions: PluginPermission[];
}

interface InstalledPluginLibraryProps {
  plugins: LibraryPluginViewModel[];
  onSubmit: (submission: AddIntegrationSubmission) => Promise<void>;
}

/**
 * Library screen for the "Add integration" dialog: lists eligible catalog
 * plugins, then walks through a lightweight mock configure step (connection
 * name + permission checkboxes) before creating the connection.
 */
export function InstalledPluginLibrary({ plugins, onSubmit }: InstalledPluginLibraryProps) {
  const [selected, setSelected] = useState<LibraryPluginViewModel | null>(null);
  const [connectionName, setConnectionName] = useState("");
  const [permissions, setPermissions] = useState<Set<PluginPermission>>(
    new Set(DEFAULT_CHECKED_PERMISSIONS),
  );
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function startConfigure(plugin: LibraryPluginViewModel) {
    setSelected(plugin);
    setConnectionName(plugin.name);
    setPermissions(new Set(DEFAULT_CHECKED_PERMISSIONS));
    setError(null);
  }

  function togglePermission(permission: PluginPermission) {
    setPermissions((current) => {
      const next = new Set(current);
      if (next.has(permission)) {
        next.delete(permission);
      } else {
        next.add(permission);
      }
      return next;
    });
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!selected) {
      return;
    }
    if (connectionName.trim().length === 0) {
      setError("Connection name is required.");
      return;
    }

    setSubmitting(true);
    setError(null);
    try {
      await onSubmit({
        installedPluginId: selected.installedPluginId,
        connectionName: connectionName.trim(),
        permissions: [...permissions],
      });
      setSelected(null);
    } catch (submitError) {
      setError(
        submitError instanceof Error
          ? submitError.message
          : "Failed to add integration.",
      );
    } finally {
      setSubmitting(false);
    }
  }

  if (selected) {
    const theme = resolveIntegrationTheme(selected.theme);
    return (
      <form onSubmit={handleSubmit}>
        <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", marginBottom: "1rem" }}>
          <IntegrationIcon icon={selected.icon} theme={theme} />
          <div>
            <strong>{selected.name}</strong>
            <div style={{ fontSize: "0.8rem", color: "var(--color-text-muted)" }}>
              {selected.publisher}
            </div>
          </div>
        </div>

        <label htmlFor="integration-connection-name" style={{ display: "block", marginBottom: "0.3rem" }}>
          Connection name
        </label>
        <Input
          id="integration-connection-name"
          value={connectionName}
          onChange={(event) => setConnectionName(event.target.value)}
          required
        />

        <fieldset style={{ marginTop: "1rem", border: "none", padding: 0 }}>
          <legend style={{ marginBottom: "0.35rem", padding: 0 }}>Permissions</legend>
          {PLUGIN_PERMISSIONS.map((permission) => (
            <label
              key={permission}
              style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.3rem" }}
            >
              <input
                type="checkbox"
                checked={permissions.has(permission)}
                onChange={() => togglePermission(permission)}
              />
              {PERMISSION_LABELS[permission]}
            </label>
          ))}
        </fieldset>

        {error ? (
          <p role="alert" style={{ color: "var(--color-danger)" }}>
            {error}
          </p>
        ) : null}

        <div style={{ display: "flex", gap: "0.5rem", marginTop: "1rem" }}>
          <Button type="submit" disabled={submitting}>
            {submitting ? "Adding\u2026" : "Add integration"}
          </Button>
          <Button type="button" onClick={() => setSelected(null)} disabled={submitting}>
            Back
          </Button>
        </div>
      </form>
    );
  }

  if (plugins.length === 0) {
    return (
      <p style={{ color: "var(--color-text-secondary)" }}>
        Every available integration is already configured for this project.
      </p>
    );
  }

  return (
    <div style={listStyle}>
      {plugins.map((plugin) => {
        const theme = resolveIntegrationTheme(plugin.theme);
        return (
          <div key={plugin.installedPluginId} style={itemStyle}>
            <IntegrationIcon icon={plugin.icon} theme={theme} />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div>
                <strong>{plugin.name}</strong>
                {plugin.badge ? (
                  <span style={badgeStyle}>
                    {plugin.badge === "built-in" ? "Built-in" : "Official"}
                  </span>
                ) : null}
              </div>
              <div style={{ fontSize: "0.8rem", color: "var(--color-text-muted)" }}>
                {plugin.publisher} &middot; v{plugin.version}
              </div>
              {plugin.description ? (
                <p style={{ margin: "0.3rem 0 0", color: "var(--color-text-secondary)" }}>
                  {plugin.description}
                </p>
              ) : null}
            </div>
            <Button onClick={() => startConfigure(plugin)}>
              {plugin.existingConnectionCount > 0 ? "Add" : "Configure"}
            </Button>
          </div>
        );
      })}
    </div>
  );
}
