export type DeploymentStatus =
  | "pending"
  | "building"
  | "ready"
  | "failed"
  | "cancelled";

export interface Deployment {
  id: string;
  projectId: string;
  environmentId: string;
  integrationId: string;
  status: DeploymentStatus;
  url?: string;
  createdAt: string;
  updatedAt: string;
}
