import { useState } from "react";

import { ProjectForm } from "./ProjectForm.js";
import { ProjectOverviewScreen } from "./ProjectOverviewScreen.js";
import { ProjectsListScreen } from "./ProjectsListScreen.js";
import { useProjects } from "./ProjectsContext.js";

type ProjectsView =
  | { kind: "list" }
  | { kind: "create" }
  | { kind: "overview"; projectId: string };

export function ProjectsFeature() {
  const { createProject } = useProjects();
  const [view, setView] = useState<ProjectsView>({ kind: "list" });

  if (view.kind === "create") {
    return (
      <section>
        <h2 style={{ marginTop: 0 }}>Create project</h2>
        <ProjectForm
          submitLabel="Create project"
          onCancel={() => setView({ kind: "list" })}
          onSubmit={async (input) => {
            const project = await createProject(input);
            setView({ kind: "overview", projectId: project.id });
          }}
        />
      </section>
    );
  }

  if (view.kind === "overview") {
    return (
      <ProjectOverviewScreen
        projectId={view.projectId}
        onBack={() => setView({ kind: "list" })}
      />
    );
  }

  return (
    <ProjectsListScreen
      onCreate={() => setView({ kind: "create" })}
      onOpen={(projectId) => setView({ kind: "overview", projectId })}
    />
  );
}
