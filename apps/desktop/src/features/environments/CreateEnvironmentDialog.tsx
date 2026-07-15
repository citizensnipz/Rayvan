import {
  useEffect,
  useId,
  useRef,
  useState,
  type CSSProperties,
  type FormEvent,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import type { EnvironmentColorToken, EnvironmentIconToken, EnvironmentKind } from "@rayvan/core";
import { Button, Input, TextArea } from "@rayvan/ui";

import { ENVIRONMENT_PRESETS } from "./view-models.js";

const overlayStyle: CSSProperties = {
  position: "fixed",
  inset: 0,
  background: "var(--color-overlay)",
  display: "grid",
  placeItems: "center",
  padding: "1.5rem",
  zIndex: 40,
};

const dialogStyle: CSSProperties = {
  width: "min(34rem, 100%)",
  maxHeight: "90vh",
  overflowY: "auto",
  background: "var(--color-surface)",
  borderRadius: "10px",
  border: "1px solid var(--color-border)",
  padding: "1.5rem",
  boxShadow: "var(--shadow-dialog)",
};

const fieldGroupStyle: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: "0.35rem",
  marginBottom: "0.9rem",
};

export interface CreateEnvironmentSubmission {
  name: string;
  kind: EnvironmentKind;
  description?: string;
  presentation: {
    color: EnvironmentColorToken;
    icon: EnvironmentIconToken;
  };
}

interface CreateEnvironmentDialogProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (submission: CreateEnvironmentSubmission) => Promise<void>;
}

const COLOR_OPTIONS: EnvironmentColorToken[] = [
  "neutral",
  "blue",
  "green",
  "amber",
  "rose",
  "violet",
  "cyan",
];

function defaultPresentation(kind: EnvironmentKind): {
  color: EnvironmentColorToken;
  icon: EnvironmentIconToken;
} {
  switch (kind) {
    case "development":
      return { color: "blue", icon: "development" };
    case "preview":
      return { color: "violet", icon: "preview" };
    case "staging":
      return { color: "amber", icon: "staging" };
    case "production":
      return { color: "rose", icon: "production" };
    case "test":
      return { color: "cyan", icon: "test" };
    case "local":
      return { color: "neutral", icon: "local" };
    case "custom":
      return { color: "neutral", icon: "custom" };
  }
}

export function CreateEnvironmentDialog({
  open,
  onClose,
  onSubmit,
}: CreateEnvironmentDialogProps) {
  const titleId = useId();
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const [presetId, setPresetId] = useState("development");
  const [name, setName] = useState("Development");
  const [kind, setKind] = useState<EnvironmentKind>("development");
  const [description, setDescription] = useState(
    ENVIRONMENT_PRESETS[0]?.description ?? "",
  );
  const [color, setColor] = useState<EnvironmentColorToken>("blue");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) {
      return;
    }
    const preset = ENVIRONMENT_PRESETS[0]!;
    setPresetId(preset.id);
    setName(preset.name);
    setKind(preset.kind);
    setDescription(preset.description);
    setColor(defaultPresentation(preset.kind).color);
    setError(null);
    setSubmitting(false);
    const previous = document.activeElement as HTMLElement | null;
    closeButtonRef.current?.focus();

    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      previous?.focus?.();
    };
  }, [open, onClose]);

  if (!open) {
    return null;
  }

  function applyPreset(id: string) {
    const preset = ENVIRONMENT_PRESETS.find((item) => item.id === id);
    if (!preset) {
      return;
    }
    setPresetId(id);
    setName(preset.name);
    setKind(preset.kind);
    setDescription(preset.description);
    setColor(defaultPresentation(preset.kind).color);
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await onSubmit({
        name: name.trim(),
        kind,
        description: description.trim() || undefined,
        presentation: {
          color,
          icon: defaultPresentation(kind).icon,
        },
      });
      onClose();
    } catch (submitError) {
      setError(
        submitError instanceof Error
          ? submitError.message
          : "Failed to create environment.",
      );
    } finally {
      setSubmitting(false);
    }
  }

  function handleDialogKeyDown(event: ReactKeyboardEvent<HTMLDivElement>) {
    if (event.key !== "Tab" || !dialogRef.current) {
      return;
    }
    const focusable = dialogRef.current.querySelectorAll<HTMLElement>(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    );
    if (focusable.length === 0) {
      return;
    }
    const first = focusable[0]!;
    const last = focusable[focusable.length - 1]!;
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  return (
    <div style={overlayStyle}>
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        style={dialogStyle}
        onKeyDown={handleDialogKeyDown}
      >
        <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem" }}>
          <h2 id={titleId} style={{ margin: 0 }}>
            Create environment
          </h2>
          <button
            ref={closeButtonRef}
            type="button"
            aria-label="Close dialog"
            onClick={onClose}
            style={{
              border: "none",
              background: "transparent",
              cursor: "pointer",
              fontSize: "1.1rem",
              color: "var(--color-text-muted)",
            }}
          >
            &times;
          </button>
        </div>
        <p style={{ color: "var(--color-text-secondary)", marginTop: "0.5rem" }}>
          Environments are saved locally. Sync with integrations is optional and read-only.
        </p>

        <form onSubmit={(event) => void handleSubmit(event)} style={{ marginTop: "1rem" }}>
          <div style={fieldGroupStyle}>
            <label htmlFor="env-preset">Preset</label>
            <select
              id="env-preset"
              value={presetId}
              onChange={(event) => applyPreset(event.target.value)}
              style={{
                padding: "0.5rem 0.75rem",
                borderRadius: "6px",
                border: "1px solid var(--color-border-strong)",
                background: "var(--color-surface)",
                color: "var(--color-text)",
              }}
            >
              {ENVIRONMENT_PRESETS.map((preset) => (
                <option key={preset.id} value={preset.id}>
                  {preset.name}
                </option>
              ))}
            </select>
          </div>

          <div style={fieldGroupStyle}>
            <label htmlFor="env-name">Name</label>
            <Input
              id="env-name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              required
            />
          </div>

          <div style={fieldGroupStyle}>
            <label htmlFor="env-kind">Type</label>
            <select
              id="env-kind"
              value={kind}
              onChange={(event) => {
                const next = event.target.value as EnvironmentKind;
                setKind(next);
                setColor(defaultPresentation(next).color);
              }}
              style={{
                padding: "0.5rem 0.75rem",
                borderRadius: "6px",
                border: "1px solid var(--color-border-strong)",
                background: "var(--color-surface)",
                color: "var(--color-text)",
              }}
            >
              {ENVIRONMENT_PRESETS.map((preset) => (
                <option key={preset.kind} value={preset.kind}>
                  {preset.name}
                </option>
              ))}
            </select>
          </div>

          <div style={fieldGroupStyle}>
            <label htmlFor="env-description">Description</label>
            <TextArea
              id="env-description"
              value={description}
              onChange={(event) => setDescription(event.target.value)}
            />
          </div>

          <div style={fieldGroupStyle}>
            <label htmlFor="env-color">Colour</label>
            <select
              id="env-color"
              value={color}
              onChange={(event) => setColor(event.target.value as EnvironmentColorToken)}
              style={{
                padding: "0.5rem 0.75rem",
                borderRadius: "6px",
                border: "1px solid var(--color-border-strong)",
                background: "var(--color-surface)",
                color: "var(--color-text)",
              }}
            >
              {COLOR_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </div>

          {error ? (
            <div role="alert" style={{ color: "var(--color-danger)", marginBottom: "0.75rem" }}>
              {error}
            </div>
          ) : null}

          <div style={{ display: "flex", gap: "0.5rem", justifyContent: "flex-end" }}>
            <Button type="button" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={submitting || name.trim().length === 0}>
              {submitting ? "Saving…" : "Create environment"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
