import type { TextareaHTMLAttributes, InputHTMLAttributes } from "react";

const fieldStyle = {
  width: "100%",
  padding: "0.5rem 0.75rem",
  borderRadius: "6px",
  border: "1px solid var(--color-border-strong)",
  background: "var(--color-surface)",
  color: "var(--color-text)",
  fontSize: "0.95rem",
  boxSizing: "border-box" as const,
};

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input style={fieldStyle} {...props} />;
}

export function TextArea(
  props: TextareaHTMLAttributes<HTMLTextAreaElement>,
) {
  return (
    <textarea
      style={{ ...fieldStyle, minHeight: "6rem", resize: "vertical" }}
      {...props}
    />
  );
}

export function StatusBadge({
  status,
}: {
  status: "active" | "archived";
}) {
  const isArchived = status === "archived";
  return (
    <span
      style={{
        display: "inline-block",
        padding: "0.15rem 0.5rem",
        borderRadius: "999px",
        fontSize: "0.75rem",
        fontWeight: 600,
        color: isArchived ? "#9a3412" : "#166534",
        background: isArchived ? "#ffedd5" : "#dcfce7",
      }}
    >
      {isArchived ? "Archived" : "Active"}
    </span>
  );
}
