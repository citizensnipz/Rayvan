import {
  useEffect,
  useId,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import { Button } from "@rayvan/ui";

import { AddPluginFromFile } from "./AddPluginFromFile.js";
import {
  InstalledPluginLibrary,
  type AddIntegrationSubmission,
} from "./InstalledPluginLibrary.js";
import type { LibraryPluginViewModel } from "./view-models.js";

type DialogScreen = "choose" | "library" | "file";

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

const choiceRowStyle: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: "0.75rem",
  marginTop: "1rem",
};

interface AddIntegrationDialogProps {
  open: boolean;
  onClose: () => void;
  plugins: LibraryPluginViewModel[];
  onSubmit: (submission: AddIntegrationSubmission) => Promise<void>;
}

export function AddIntegrationDialog({
  open,
  onClose,
  plugins,
  onSubmit,
}: AddIntegrationDialogProps) {
  const [screen, setScreen] = useState<DialogScreen>("choose");
  const titleId = useId();
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) {
      return;
    }
    setScreen("choose");
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
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (!first || !last) {
      return;
    }
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  return (
    <div
      style={overlayStyle}
      role="presentation"
      onClick={onClose}
    >
      <div
        ref={dialogRef}
        style={dialogStyle}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(event) => event.stopPropagation()}
        onKeyDown={handleDialogKeyDown}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2 id={titleId} style={{ marginTop: 0, marginBottom: 0 }}>
            Add integration
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

        {screen === "choose" ? (
          <div style={choiceRowStyle}>
            <p style={{ margin: 0, color: "var(--color-text-secondary)" }}>
              Choose how you want to add an integration to this project.
            </p>
            <Button onClick={() => setScreen("library")}>Add from library</Button>
            <Button onClick={() => setScreen("file")}>Add from file</Button>
          </div>
        ) : null}

        {screen === "library" ? (
          <div style={{ marginTop: "1rem" }}>
            <Button type="button" onClick={() => setScreen("choose")}>
              &larr; Back
            </Button>
            <div style={{ marginTop: "1rem" }}>
              <InstalledPluginLibrary plugins={plugins} onSubmit={onSubmit} />
            </div>
          </div>
        ) : null}

        {screen === "file" ? (
          <div style={{ marginTop: "1rem" }}>
            <Button type="button" onClick={() => setScreen("choose")}>
              &larr; Back
            </Button>
            <div style={{ marginTop: "1rem" }}>
              <AddPluginFromFile />
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
