import type { CSSProperties, PropsWithChildren, ReactNode } from "react";

export interface AppShellProps extends PropsWithChildren {
  topNav?: ReactNode;
  sidebar?: ReactNode;
}

const shellStyle: CSSProperties = {
  minHeight: "100vh",
  display: "grid",
  gridTemplateRows: "auto 1fr",
  fontFamily: '"Figtree", system-ui, sans-serif',
  color: "var(--color-text)",
  background: "var(--color-bg)",
};

const bodyStyle: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "14rem 1fr",
  minHeight: 0,
};

const mainStyle: CSSProperties = {
  padding: "1.5rem 2rem",
  overflow: "auto",
  minWidth: 0,
};

export function AppShell({ topNav, sidebar, children }: AppShellProps) {
  return (
    <div style={shellStyle}>
      {topNav}
      <div style={bodyStyle}>
        {sidebar}
        <main style={mainStyle}>{children}</main>
      </div>
    </div>
  );
}
