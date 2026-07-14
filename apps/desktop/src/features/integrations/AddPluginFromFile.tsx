import { useState, type ChangeEvent } from "react";
import { Button } from "@rayvan/ui";

interface ParsedManifestMetadata {
  id: string;
  name: string;
  version: string;
  publisher: string;
}

type FileParseResult =
  | { kind: "manifest"; metadata: ParsedManifestMetadata }
  | { kind: "unsupported"; reason: string };

function looksLikeManifestMetadata(value: unknown): value is ParsedManifestMetadata {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const record = value as Record<string, unknown>;
  return (
    typeof record.id === "string" &&
    typeof record.name === "string" &&
    typeof record.version === "string" &&
    typeof record.publisher === "string"
  );
}

function hasSupportedExtension(fileName: string): boolean {
  const lower = fileName.toLowerCase();
  return lower.endsWith(".rayvan-plugin") || lower.endsWith(".json");
}

/** `File.text()` is not implemented by every runtime (notably jsdom); `FileReader` is. */
function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(reader.error ?? new Error("Failed to read file"));
    reader.readAsText(file);
  });
}

/**
 * Parses a `.rayvan-plugin` file's manifest metadata only, as text/JSON.
 * Never evaluates, imports, or otherwise executes file contents — this
 * screen exists purely to demonstrate the intended UX for local development
 * builds where external plugin execution is disabled.
 */
async function parsePluginFile(file: File): Promise<FileParseResult> {
  if (!hasSupportedExtension(file.name)) {
    return {
      kind: "unsupported",
      reason:
        "Unsupported file type. Only .rayvan-plugin (or metadata .json) files are supported.",
    };
  }

  const text = await readFileAsText(file);
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return {
      kind: "unsupported",
      reason: "File could not be parsed as plugin manifest metadata.",
    };
  }

  if (!looksLikeManifestMetadata(parsed)) {
    return {
      kind: "unsupported",
      reason: "File does not contain recognizable plugin manifest metadata.",
    };
  }

  return { kind: "manifest", metadata: parsed };
}

export function AddPluginFromFile() {
  const [result, setResult] = useState<FileParseResult | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  async function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return;
    }
    setFileName(file.name);
    const parseResult = await parsePluginFile(file);
    setResult(parseResult);
  }

  return (
    <div>
      <p style={{ color: "var(--color-text-secondary)" }}>
        Add a plugin from a local <code>.rayvan-plugin</code> file. Only manifest
        metadata is read &mdash; no plugin code is executed.
      </p>

      <Button
        type="button"
        onClick={() => document.getElementById("plugin-file-input")?.click()}
      >
        Choose file&hellip;
      </Button>
      <input
        id="plugin-file-input"
        type="file"
        aria-label="Choose plugin file"
        accept=".rayvan-plugin,application/json"
        style={{
          position: "absolute",
          width: 1,
          height: 1,
          padding: 0,
          margin: 0,
          overflow: "hidden",
          clip: "rect(0,0,0,0)",
          whiteSpace: "nowrap",
          border: 0,
        }}
        onChange={handleFileChange}
      />

      {fileName ? (
        <p style={{ marginTop: "0.75rem", color: "var(--color-text-muted)" }}>
          Selected: {fileName}
        </p>
      ) : null}

      {result?.kind === "manifest" ? (
        <div role="status" style={{ marginTop: "0.75rem" }}>
          <p>
            Detected plugin <strong>{result.metadata.name}</strong> (v
            {result.metadata.version}) by {result.metadata.publisher}.
          </p>
          <p style={{ color: "var(--color-text-secondary)" }}>
            External plugin execution is not enabled in this build. This plugin
            cannot be installed or run from a file yet.
          </p>
        </div>
      ) : null}

      {result?.kind === "unsupported" ? (
        <p role="alert" style={{ marginTop: "0.75rem", color: "var(--color-danger)" }}>
          {result.reason}
        </p>
      ) : null}
    </div>
  );
}
