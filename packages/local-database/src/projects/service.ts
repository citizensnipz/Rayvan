import type { Project } from "@rayvan/core";

import type { ProjectRepository } from "./repository.js";

export class ProjectService {
  constructor(private readonly repository: ProjectRepository) {}

  list(options?: { includeArchived?: boolean }): Promise<Project[]> {
    return this.repository.list(options);
  }

  getById(id: string): Promise<Project | null> {
    return this.repository.getById(id);
  }

  create(input: { name: string; description?: string }): Promise<Project> {
    return this.repository.create(input);
  }

  update(
    id: string,
    input: { name?: string; description?: string },
  ): Promise<Project> {
    return this.repository.update(id, input);
  }

  archive(id: string): Promise<Project> {
    return this.repository.setArchived(id, true);
  }

  restore(id: string): Promise<Project> {
    return this.repository.setArchived(id, false);
  }

  delete(id: string): Promise<void> {
    return this.repository.delete(id);
  }
}
