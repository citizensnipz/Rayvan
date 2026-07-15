import { useResolvedCurrentProject } from "../../app/CurrentProjectContext.js";
import { IntegrationsProvider } from "./IntegrationsContext.js";
import { IntegrationsWorkspace } from "./IntegrationsWorkspace.js";

export function IntegrationsPage() {
  const { currentProjectId } = useResolvedCurrentProject();

  return (
    <IntegrationsProvider>
      <IntegrationsWorkspace projectId={currentProjectId} />
    </IntegrationsProvider>
  );
}
