import { useResolvedCurrentProject } from "../../app/CurrentProjectContext.js";
import { FindingsProvider } from "./FindingsContext.js";
import { FindingsWorkspace } from "./FindingsWorkspace.js";

/**
 * Findings workspace entry.
 *
 * How to open development fixtures:
 * 1. Select a project in the desktop app
 * 2. Open the Findings section in the sidebar
 * 3. The dev gateway auto-seeds FindingRecords via ensureProjectSeeded
 * 4. Use “Scan project” to run evaluateProject against the seeded context
 */
export function FindingsPage() {
  const { currentProjectId } = useResolvedCurrentProject();

  return (
    <FindingsProvider>
      <FindingsWorkspace projectId={currentProjectId} />
    </FindingsProvider>
  );
}
