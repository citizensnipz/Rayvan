import type { ButtonHTMLAttributes, PropsWithChildren } from "react";

export interface NavItemProps
  extends PropsWithChildren<ButtonHTMLAttributes<HTMLButtonElement>> {
  active?: boolean;
}

export function NavItem({
  children,
  active = false,
  disabled,
  ...props
}: NavItemProps) {
  return (
    <button
      type="button"
      {...props}
      disabled={disabled}
      aria-current={active ? "page" : undefined}
      style={{
        display: "block",
        width: "100%",
        textAlign: "left",
        padding: "0.55rem 0.75rem",
        borderRadius: "6px",
        border: "none",
        background: active ? "var(--color-nav-active)" : "transparent",
        color: disabled
          ? "var(--color-text-muted)"
          : active
            ? "var(--color-text)"
            : "var(--color-text-secondary)",
        fontWeight: active ? 600 : 500,
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.7 : 1,
        ...props.style,
      }}
    >
      {children}
    </button>
  );
}
