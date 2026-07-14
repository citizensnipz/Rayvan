import type { IntegrationId, ProjectId } from "../ids/index.js";

export type IntegrationStatus = "connected" | "disconnected" | "error";

export interface Integration {
  id: IntegrationId;
  projectId: ProjectId;
  pluginId: string;
  displayName: string;
  status: IntegrationStatus;
}
