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
        border: "1px solid #cbd5e1",
        background: props.disabled ? "#f1f5f9" : "#ffffff",
        color: props.disabled ? "#94a3b8" : "#0f172a",
        cursor: props.disabled ? "not-allowed" : "pointer",
      }}
      {...props}
    >
      {children}
    </button>
  );
}
