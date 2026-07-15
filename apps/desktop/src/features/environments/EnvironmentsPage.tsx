import { useResolvedCurrentProject } from "../../app/CurrentProjectContext.js";
import { EnvironmentsProvider } from "./EnvironmentsContext.js";
import { EnvironmentsWorkspace } from "./EnvironmentsWorkspace.js";

export function EnvironmentsPage() {
  const { currentProjectId } = useResolvedCurrentProject();

  return (
    <EnvironmentsProvider>
      <EnvironmentsWorkspace projectId={currentProjectId} />
    </EnvironmentsProvider>
  );
}
