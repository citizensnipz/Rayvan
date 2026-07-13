import type { Project } from "@rayvan/core";

export interface ProjectRepository {
  list(options?: { includeArchived?: boolean }): Promise<Project[]>;
  getById(id: string): Promise<Project | null>;
  create(input: { name: string; description?: string }): Promise<Project>;
  update(
    id: string,
    input: { name?: string; description?: string },
  ): Promise<Project>;
  setArchived(id: string, archived: boolean): Promise<Project>;
  delete(id: string): Promise<void>;
}
