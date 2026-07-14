import { useResolvedCurrentProject } from "../../app/CurrentProjectContext.js";
import { useProjects } from "./ProjectsContext.js";
import { ProjectOverviewScreen } from "./ProjectOverviewScreen.js";
import { ProjectsListScreen } from "./ProjectsListScreen.js";

export function OverviewPage() {
  const { loading, error, projects } = useProjects();
  const {
    currentProjectId,
    currentProject,
    setCurrentProjectId,
    sessionReady,
  } = useResolvedCurrentProject();

  if (loading || !sessionReady) {
    return (
      <p style={{ margin: 0, color: "var(--color-text-secondary)" }}>
        Loading projects...
      </p>
    );
  }

  if (error) {
    return (
      <p style={{ margin: 0, color: "var(--color-danger)" }} role="alert">
        {error}
      </p>
    );
  }

  if (currentProjectId && currentProject) {
    return <ProjectOverviewScreen projectId={currentProjectId} />;
  }

  if (projects.length > 0) {
    return (
      <ProjectsListScreen
        onOpen={(projectId) => setCurrentProjectId(projectId)}
      />
    );
  }

  return (
    <p style={{ margin: 0, color: "var(--color-text-secondary)" }}>
      Create a new project to get started
    </p>
  );
}
