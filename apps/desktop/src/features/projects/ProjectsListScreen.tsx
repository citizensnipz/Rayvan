import { Button, EmptyState, StatusBadge } from "@rayvan/ui";
import type { Project } from "@rayvan/core";

import { formatDateTime } from "../../lib/format.js";
import { useProjects } from "./ProjectsContext.js";

interface ProjectsListScreenProps {
  onCreate: () => void;
  onOpen: (projectId: string) => void;
}

export function ProjectsListScreen({
  onCreate,
  onOpen,
}: ProjectsListScreenProps) {
  const {
    projects,
    loading,
    error,
    includeArchived,
    setIncludeArchived,
    archiveProject,
    restoreProject,
    deleteProject,
  } = useProjects();

  async function handleArchiveToggle(project: Project) {
    if (project.status === "archived") {
      await restoreProject(project.id);
      return;
    }
    await archiveProject(project.id);
  }

  async function handleDelete(project: Project) {
    const confirmed = window.confirm(
      `Permanently delete "${project.name}"? This cannot be undone.`,
    );
    if (!confirmed) {
      return;
    }
    await deleteProject(project.id);
  }

  return (
    <section>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: "1rem",
          flexWrap: "wrap",
        }}
      >
        <div>
          <h2 style={{ margin: 0 }}>Projects</h2>
          <p style={{ margin: "0.35rem 0 0", color: "#475569" }}>
            Persistent workspaces for the software you operate.
          </p>
        </div>
        <Button onClick={onCreate}>Create project</Button>
      </div>

      <label
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "0.5rem",
          marginTop: "1rem",
          color: "#334155",
        }}
      >
        <input
          type="checkbox"
          checked={includeArchived}
          onChange={(event) => setIncludeArchived(event.target.checked)}
        />
        Show archived projects
      </label>

      {loading ? (
        <p style={{ marginTop: "1.5rem", color: "#475569" }}>Loading projects...</p>
      ) : null}

      {error ? (
        <p style={{ marginTop: "1.5rem", color: "#b91c1c" }} role="alert">
          {error}
        </p>
      ) : null}

      {!loading && !error && projects.length === 0 ? (
        <EmptyState
          title="No projects yet."
          description="Create a project to start organizing environments and resources."
        />
      ) : null}

      {!loading && !error && projects.length > 0 ? (
        <ul
          style={{
            listStyle: "none",
            margin: "1.5rem 0 0",
            padding: 0,
            display: "grid",
            gap: "0.75rem",
          }}
        >
          {projects.map((project) => (
            <li
              key={project.id}
              style={{
                padding: "1rem",
                borderRadius: "8px",
                border: "1px solid #e2e8f0",
                background: "#ffffff",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  gap: "1rem",
                  alignItems: "flex-start",
                  flexWrap: "wrap",
                }}
              >
                <div>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem",
                      flexWrap: "wrap",
                    }}
                  >
                    <strong>{project.name}</strong>
                    <StatusBadge status={project.status} />
                  </div>
                  {project.description ? (
                    <p style={{ margin: "0.5rem 0 0", color: "#475569" }}>
                      {project.description}
                    </p>
                  ) : null}
                  <p style={{ margin: "0.5rem 0 0", color: "#64748b", fontSize: "0.875rem" }}>
                    Updated {formatDateTime(project.updatedAt)}
                  </p>
                </div>

                <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                  <Button onClick={() => onOpen(project.id)}>Open</Button>
                  <Button onClick={() => void handleArchiveToggle(project)}>
                    {project.status === "archived" ? "Restore" : "Archive"}
                  </Button>
                  <Button onClick={() => void handleDelete(project)}>Delete</Button>
                </div>
              </div>
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
