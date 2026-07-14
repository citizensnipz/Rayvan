import type { ButtonHTMLAttributes, PropsWithChildren } from "react";

export function Button({
  children,
  ...props
}: PropsWithChildren<ButtonHTMLAttributes<HTMLButtonElement>>) {
  return (
    <button
      type="button"
      style={{
        padding: "0.5rem 1rem",
        borderRadius: "6px",
        border: "1px solid var(--color-border-strong)",
        background: props.disabled
          ? "var(--color-surface-muted)"
          : "var(--color-surface)",
        color: props.disabled
          ? "var(--color-text-muted)"
          : "var(--color-text)",
        cursor: props.disabled ? "not-allowed" : "pointer",
      }}
      {...props}
    >
      {children}
    </button>
  );
}
