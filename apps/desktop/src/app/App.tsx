import { AppShell } from "@rayvan/ui";

import { ProjectsFeature } from "../features/projects/index.js";
import { ProjectsProvider } from "../features/projects/ProjectsContext.js";

export function App() {
  return (
    <AppShell>
      <header>
        <h1 style={{ margin: 0, fontSize: "2rem" }}>Rayvan</h1>
        <p style={{ margin: "0.5rem 0 0", color: "#475569" }}>
          Local-first infrastructure control plane.
        </p>
      </header>

      <div style={{ marginTop: "2rem" }}>
        <ProjectsProvider>
          <ProjectsFeature />
        </ProjectsProvider>
      </div>
    </AppShell>
  );
}
