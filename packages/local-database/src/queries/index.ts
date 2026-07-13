export interface ListProjectsQuery {
  workspaceId: string;
}

export function buildListProjectsQuery(workspaceId: string): ListProjectsQuery {
  return { workspaceId };
}
