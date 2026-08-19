import { useState, type ChangeEvent, type FormEvent } from "react";
import { Button } from "@rayvan/ui";

import { desktopDaemon } from "../../lib/daemon/client.js";

export interface InstalledPluginFromFile {
  pluginId: string;
  version: string;
  trustLabel?: string;
}

interface AddPluginFromFileProps {
  onInstalled?: (installed: InstalledPluginFromFile) => void;
}

/**
 * Installs a platform-specific `.rayvan-plugin` package via the daemon.
 * The webview never imports or executes plugin code — only a filesystem path
 * is sent to `plugins.installFromPath`.
 */
export function AddPluginFromFile({ onInstalled }: AddPluginFromFileProps) {
  const [path, setPath] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<InstalledPluginFromFile | null>(null);

  async function handleInstall(event: FormEvent) {
    event.preventDefault();
    const trimmed = path.trim();
    if (!trimmed) {
      setError("Enter the absolute path to a .rayvan-plugin file.");
      return;
    }
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const installed = await desktopDaemon.installPluginFromPath(trimmed);
      const next = {
        pluginId: installed.pluginId,
        version: installed.version,
        trustLabel: installed.trustLabel,
      };
      setResult(next);
      onInstalled?.(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  function handlePathChange(event: ChangeEvent<HTMLInputElement>) {
    setPath(event.target.value);
    setError(null);
    setResult(null);
  }

  return (
    <div>
      <p style={{ color: "var(--color-text-secondary)" }}>
        Install a platform-specific <code>.rayvan-plugin</code> package through
        the Rayvan daemon. Plugin code never runs in the desktop renderer.
        Unsigned development packages require{" "}
        <code>RAYVAN_ALLOW_UNSIGNED_PLUGINS=1</code> (enabled automatically for
        desktop debug builds).
      </p>

      <form onSubmit={handleInstall} style={{ marginTop: "1rem" }}>
        <label
          htmlFor="plugin-package-path"
          style={{ display: "block", marginBottom: "0.35rem" }}
        >
          Package path
        </label>
        <input
          id="plugin-package-path"
          type="text"
          value={path}
          onChange={handlePathChange}
          placeholder="C:\\path\\to\\io.rayvan.github-0.1.0-….rayvan-plugin"
          style={{
            width: "100%",
            padding: "0.5rem 0.65rem",
            borderRadius: "6px",
            border: "1px solid var(--color-border)",
            background: "var(--color-surface-muted)",
            color: "var(--color-text)",
          }}
        />
        <div style={{ marginTop: "0.75rem" }}>
          <Button type="submit" disabled={busy}>
            {busy ? "Installing…" : "Install package"}
          </Button>
        </div>
      </form>

      {result ? (
        <div role="status" style={{ marginTop: "0.75rem" }}>
          <p>
            Installed <strong>{result.pluginId}</strong> v{result.version}.
          </p>
          {result.trustLabel ? (
            <p style={{ color: "var(--color-text-secondary)" }}>
              Trust: {result.trustLabel}
            </p>
          ) : null}
          <p style={{ color: "var(--color-text-secondary)" }}>
            The package is on this machine but not connected to the project yet.
          </p>
          {onInstalled ? (
            <div style={{ marginTop: "0.75rem" }}>
              <Button type="button" onClick={() => onInstalled(result)}>
                Set up {result.pluginId.includes("github") ? "GitHub" : "plugin"}
              </Button>
            </div>
          ) : (
            <p style={{ color: "var(--color-text-secondary)" }}>
              Next: use Add from library to authenticate and create a project
              connection.
            </p>
          )}
        </div>
      ) : null}

      {error ? (
        <p role="alert" style={{ marginTop: "0.75rem", color: "var(--color-danger)" }}>
          {error}
        </p>
      ) : null}
    </div>
  );
}
