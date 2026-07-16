import { useMemo, useState, type CSSProperties } from "react";
import type { ProjectId } from "@rayvan/core";
import { Button } from "@rayvan/ui";

import rayvanLogoMed from "../../assets/brand/rayvan-logo-med.png";
import { useProjects } from "../../features/projects/ProjectsContext.js";
import { useDaemonConnection } from "../../lib/daemon/index.js";
import { useResolvedCurrentProject } from "../CurrentProjectContext.js";
import { useAppNavigation } from "../navigation/AppNavigationContext.js";

const navStyle: CSSProperties = {
  display: "flex",
  justifyContent: "flex-start",
  alignItems: "center",
  gap: "0.75rem",
  padding: "0.85rem 1.5rem",
  borderBottom: "1px solid var(--color-border)",
  background: "var(--color-surface)",
};

const brandStyle: CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: "0.55rem",
  margin: 0,
};

const brandLogoStyle: CSSProperties = {
  display: "block",
  height: "1.75rem",
  width: "auto",
};

const brandNameStyle: CSSProperties = {
  margin: 0,
  fontFamily: '"Quicksand", sans-serif',
  fontSize: "1.25rem",
  fontWeight: 700,
  letterSpacing: "-0.02em",
};

const menuStyle: CSSProperties = {
  position: "absolute",
  left: 0,
  top: "calc(100% + 0.35rem)",
  minWidth: "14rem",
  margin: 0,
  padding: "0.35rem",
  listStyle: "none",
  background: "var(--color-surface)",
  border: "1px solid var(--color-border)",
  borderRadius: "8px",
  boxShadow: "var(--shadow-menu)",
  zIndex: 30,
};

const menuItemStyle: CSSProperties = {
  display: "block",
  width: "100%",
  textAlign: "left",
  padding: "0.5rem 0.75rem",
  border: "none",
  borderRadius: "6px",
  background: "transparent",
  color: "var(--color-text)",
  cursor: "pointer",
};

const daemonBannerStyle: CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: "0.75rem",
  marginLeft: "auto",
  padding: "0.35rem 0.75rem",
  borderRadius: "8px",
  border: "1px solid var(--color-border-strong)",
  background: "var(--color-surface-muted)",
  color: "var(--color-text)",
  fontSize: "0.85rem",
  maxWidth: "28rem",
};

export function TopNav() {
  const { projects } = useProjects();
  const { currentProject, hasProject, setCurrentProjectId } =
    useResolvedCurrentProject();
  const { openCreateProject, setActiveSection } = useAppNavigation();
  const { loading, connected, lastError, reconnect } = useDaemonConnection();
  const [menuOpen, setMenuOpen] = useState(false);

  const activeProjects = useMemo(
    () => projects.filter((project) => project.status === "active"),
    [projects],
  );

  function selectProject(id: ProjectId) {
    setCurrentProjectId(id);
    setActiveSection("overview");
    setMenuOpen(false);
  }

  const selectorLabel =
    hasProject && currentProject
      ? currentProject.name
      : activeProjects.length > 0
        ? "Select project"
        : null;

  const showDaemonBanner = !loading && !connected;

  return (
    <header style={navStyle}>
      <div style={brandStyle}>
        <img
          src={rayvanLogoMed}
          alt=""
          aria-hidden="true"
          style={brandLogoStyle}
        />
        <h1 style={brandNameStyle}>Rayvan</h1>
      </div>
      {selectorLabel ? (
        <div style={{ position: "relative" }}>
          <Button
            aria-label={
              hasProject && currentProject
                ? `Current project: ${currentProject.name}`
                : "Select project"
            }
            aria-expanded={menuOpen}
            aria-haspopup="menu"
            onClick={() => setMenuOpen((open) => !open)}
          >
            {selectorLabel}
          </Button>
          {menuOpen ? (
            <ul style={menuStyle} role="menu">
              {activeProjects.map((project) => (
                <li key={project.id} role="none">
                  <button
                    type="button"
                    role="menuitem"
                    style={{
                      ...menuItemStyle,
                      fontWeight:
                        project.id === currentProject?.id ? 600 : 400,
                      background:
                        project.id === currentProject?.id
                          ? "var(--color-surface-muted)"
                          : "transparent",
                    }}
                    onClick={() => selectProject(project.id)}
                  >
                    {project.name}
                  </button>
                </li>
              ))}
              <li role="none">
                <button
                  type="button"
                  role="menuitem"
                  style={{
                    ...menuItemStyle,
                    borderTop: "1px solid var(--color-border)",
                    marginTop: "0.25rem",
                    borderRadius: "0 0 6px 6px",
                  }}
                  onClick={() => {
                    setMenuOpen(false);
                    openCreateProject();
                  }}
                >
                  + Create new project
                </button>
              </li>
            </ul>
          ) : null}
        </div>
      ) : (
        <Button aria-label="Create new project" onClick={openCreateProject}>
          + Create new project
        </Button>
      )}
      {showDaemonBanner ? (
        <div role="status" style={daemonBannerStyle}>
          <span>
            Local daemon offline
            {lastError ? `: ${lastError}` : ". Start rayvand or retry."}
          </span>
          <Button onClick={() => void reconnect()}>Retry</Button>
        </div>
      ) : null}
    </header>
  );
}
