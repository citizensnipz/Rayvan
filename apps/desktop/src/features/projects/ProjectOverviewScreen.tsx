import { useEffect, useState } from "react";
import { Button, StatusBadge } from "@rayvan/ui";
import type { Project } from "@rayvan/core";

import { formatDateTime } from "../../lib/format.js";
import { useProjects } from "./ProjectsContext.js";
import { ProjectForm } from "./ProjectForm.js";

interface ProjectOverviewScreenProps {
  projectId: string;
  onBack: () => void;
}

export function ProjectOverviewScreen({
  projectId,
  onBack,
}: ProjectOverviewScreenProps) {
  const { getProject, updateProject } = useProjects();
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);

  useEffect(() => {
    let active = true;

    async function loadProject() {
      setLoading(true);
      setError(null);
      try {
        const loaded = await getProject(projectId);
        if (!active) {
          return;
        }
        if (!loaded) {
          setError("Project not found.");
          setProject(null);
          return;
        }
        setProject(loaded);
      } catch (loadError) {
        if (!active) {
          return;
        }
        setError(
          loadError instanceof Error
            ? loadError.message
            : "Failed to load project",
        );
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    void loadProject();

    return () => {
      active = false;
    };
  }, [getProject, projectId]);

  if (loading) {
    return <p style={{ color: "#475569" }}>Loading project...</p>;
  }

  if (error || !project) {
    return (
      <section>
        <p style={{ color: "#b91c1c" }} role="alert">
          {error ?? "Project not found."}
        </p>
        <Button onClick={onBack}>Back to projects</Button>
      </section>
    );
  }

  if (editing) {
    return (
      <section>
        <h2 style={{ marginTop: 0 }}>Edit project</h2>
        <ProjectForm
          initialValues={project}
          submitLabel="Save changes"
          onCancel={() => setEditing(false)}
          onSubmit={async (input) => {
            const updated = await updateProject(project.id, input);
            setProject(updated);
            setEditing(false);
          }}
        />
      </section>
    );
  }

  return (
    <section>
      <Button onClick={onBack}>Back to projects</Button>

      <div style={{ marginTop: "1rem" }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.75rem",
            flexWrap: "wrap",
          }}
        >
          <h2 style={{ margin: 0 }}>{project.name}</h2>
          <StatusBadge status={project.status} />
        </div>

        {project.description ? (
          <p style={{ margin: "0.75rem 0 0", color: "#475569" }}>
            {project.description}
          </p>
        ) : (
          <p style={{ margin: "0.75rem 0 0", color: "#94a3b8" }}>
            No description provided.
          </p>
        )}

        <dl
          style={{
            margin: "1.25rem 0",
            display: "grid",
            gap: "0.5rem",
            color: "#334155",
          }}
        >
          <div>
            <dt style={{ fontWeight: 600 }}>Created</dt>
            <dd style={{ margin: 0 }}>{formatDateTime(project.createdAt)}</dd>
          </div>
          <div>
            <dt style={{ fontWeight: 600 }}>Last updated</dt>
            <dd style={{ margin: 0 }}>{formatDateTime(project.updatedAt)}</dd>
          </div>
        </dl>

        <Button onClick={() => setEditing(true)}>Edit project</Button>

        <div
          style={{
            marginTop: "1.5rem",
            padding: "1rem",
            borderRadius: "8px",
            border: "1px dashed #cbd5e1",
            background: "#ffffff",
            color: "#475569",
          }}
        >
          Resources and environments will be added here in a future release.
        </div>
      </div>
    </section>
  );
}
