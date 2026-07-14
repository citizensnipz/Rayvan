import type { CSSProperties } from "react";
import { AppShell } from "@rayvan/ui";

import { useProjects } from "../../features/projects/ProjectsContext.js";
import { ProjectForm } from "../../features/projects/ProjectForm.js";
import { useResolvedCurrentProject } from "../CurrentProjectContext.js";
import { useAppNavigation } from "../navigation/AppNavigationContext.js";
import { SECTION_PAGES } from "../sectionRegistry.js";
import { Sidebar } from "./Sidebar.js";
import { TopNav } from "./TopNav.js";

const overlayStyle: CSSProperties = {
  position: "fixed",
  inset: 0,
  background: "var(--color-overlay)",
  display: "grid",
  placeItems: "center",
  padding: "1.5rem",
  zIndex: 40,
};

const dialogStyle: CSSProperties = {
  width: "min(32rem, 100%)",
  background: "var(--color-surface)",
  borderRadius: "10px",
  border: "1px solid var(--color-border)",
  padding: "1.5rem",
  boxShadow: "var(--shadow-dialog)",
};

export function DesktopShell() {
  const { createProject } = useProjects();
  const { setCurrentProjectId } = useResolvedCurrentProject();
  const {
    activeSection,
    setActiveSection,
    isCreatingProject,
    closeCreateProject,
  } = useAppNavigation();

  const Page = SECTION_PAGES[activeSection];

  return (
    <>
      <AppShell topNav={<TopNav />} sidebar={<Sidebar />}>
        <Page />
      </AppShell>

      {isCreatingProject ? (
        <div
          style={overlayStyle}
          role="presentation"
          onClick={closeCreateProject}
        >
          <div
            style={dialogStyle}
            role="dialog"
            aria-modal="true"
            aria-labelledby="create-project-title"
            onClick={(event) => event.stopPropagation()}
          >
            <h2 id="create-project-title" style={{ marginTop: 0 }}>
              Create new project
            </h2>
            <ProjectForm
              submitLabel="Create project"
              onCancel={closeCreateProject}
              onSubmit={async (input) => {
                const project = await createProject(input);
                setCurrentProjectId(project.id);
                setActiveSection("overview");
                closeCreateProject();
              }}
            />
          </div>
        </div>
      ) : null}
    </>
  );
}
