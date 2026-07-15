import type {
  Environment,
  EnvironmentKind,
  EnvironmentPresentation,
  EnvironmentStatus,
} from "@rayvan/core";

export interface CreateEnvironmentInput {
  projectId: string;
  name: string;
  slug: string;
  kind: EnvironmentKind;
  description?: string;
  presentation?: EnvironmentPresentation;
  status?: EnvironmentStatus;
}

export interface UpdateEnvironmentInput {
  name?: string;
  slug?: string;
  description?: string;
  kind?: EnvironmentKind;
  presentation?: EnvironmentPresentation;
  status?: EnvironmentStatus;
}

export interface EnvironmentRepository {
  listByProjectId(
    projectId: string,
    options?: { includeArchived?: boolean },
  ): Promise<Environment[]>;
  getById(id: string): Promise<Environment | null>;
  getByProjectAndSlug(
    projectId: string,
    slug: string,
  ): Promise<Environment | null>;
  getByProjectAndName(
    projectId: string,
    name: string,
  ): Promise<Environment | null>;
  create(input: CreateEnvironmentInput): Promise<Environment>;
  update(id: string, input: UpdateEnvironmentInput): Promise<Environment>;
  setStatus(id: string, status: EnvironmentStatus): Promise<Environment>;
  delete?(id: string): Promise<void>;
}
