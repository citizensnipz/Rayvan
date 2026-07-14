import type { CSSProperties } from "react";

import { useTheme } from "../../app/theme/ThemeContext.js";

const rowStyle: CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: "1.5rem",
  maxWidth: "28rem",
  padding: "1rem 1.25rem",
  borderRadius: "8px",
  border: "1px solid var(--color-border)",
  background: "var(--color-surface)",
};

const switchTrackStyle = (on: boolean): CSSProperties => ({
  position: "relative",
  width: "2.75rem",
  height: "1.5rem",
  borderRadius: "999px",
  border: "none",
  padding: 0,
  background: on
    ? "var(--color-switch-track-on)"
    : "var(--color-switch-track)",
  cursor: "pointer",
  flexShrink: 0,
  transition: "background 120ms ease",
});

const switchThumbStyle = (on: boolean): CSSProperties => ({
  position: "absolute",
  top: "0.15rem",
  left: on ? "calc(100% - 1.2rem)" : "0.15rem",
  width: "1.2rem",
  height: "1.2rem",
  borderRadius: "999px",
  background: "var(--color-switch-thumb)",
  boxShadow: "0 1px 3px rgba(15, 23, 42, 0.25)",
  transition: "left 120ms ease",
});

export function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const darkMode = theme === "dark";

  return (
    <section style={{ display: "grid", gap: "1.5rem" }}>
      <div>
        <h2 style={{ marginTop: 0 }}>Settings</h2>
        <p style={{ margin: "0.35rem 0 0", color: "var(--color-text-secondary)" }}>
          Appearance and preferences.
        </p>
      </div>

      <div style={rowStyle}>
        <div>
          <div style={{ fontWeight: 600 }}>Dark mode</div>
          <p
            style={{
              margin: "0.25rem 0 0",
              fontSize: "0.875rem",
              color: "var(--color-text-secondary)",
            }}
          >
            {darkMode ? "Using dark appearance" : "Using light appearance"}
          </p>
        </div>
        <button
          type="button"
          role="switch"
          aria-checked={darkMode}
          aria-label="Dark mode"
          style={switchTrackStyle(darkMode)}
          onClick={() => setTheme(darkMode ? "light" : "dark")}
        >
          <span style={switchThumbStyle(darkMode)} aria-hidden />
        </button>
      </div>
    </section>
  );
}
