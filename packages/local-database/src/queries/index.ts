export interface ListProjectsQuery {
  includeArchived?: boolean;
}

export function buildListProjectsQuery(
  includeArchived = false,
): ListProjectsQuery {
  return { includeArchived };
}
