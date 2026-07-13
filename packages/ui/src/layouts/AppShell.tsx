import type { PropsWithChildren } from "react";

export function AppShell({ children }: PropsWithChildren) {
  return (
    <div
      style={{
        minHeight: "100vh",
        fontFamily: "system-ui, sans-serif",
        color: "#0f172a",
        background: "#f8fafc",
        padding: "2rem",
      }}
    >
      {children}
    </div>
  );
}
