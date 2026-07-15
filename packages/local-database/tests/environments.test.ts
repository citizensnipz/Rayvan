import { describe, expect, it } from "vitest";

import {
  DuplicateEnvironmentNameError,
  DuplicateEnvironmentSlugError,
  EnvironmentService,
  InMemoryEnvironmentRepository,
} from "../src/environments/index.js";

function createService(): EnvironmentService {
  return new EnvironmentService(new InMemoryEnvironmentRepository());
}

describe("EnvironmentService", () => {
  it("creates a local environment", async () => {
    const service = createService();
    const environment = await service.create({
      projectId: "project-a",
      name: "Local",
      kind: "local",
    });

    expect(environment.name).toBe("Local");
    expect(environment.slug).toBe("local");
    expect(environment.kind).toBe("local");
    expect(environment.status).toBe("local_only");
    expect(environment.presentation).toEqual({
      color: "neutral",
      icon: "local",
    });
    expect(environment.projectId).toBe("project-a");
  });

  it("rejects duplicate names within a project", async () => {
    const service = createService();
    await service.create({
      projectId: "project-a",
      name: "Staging",
      kind: "staging",
    });

    await expect(
      service.create({
        projectId: "project-a",
        name: "Staging",
        kind: "staging",
        slug: "staging-2",
      }),
    ).rejects.toBeInstanceOf(DuplicateEnvironmentNameError);
  });

  it("rejects duplicate slugs within a project", async () => {
    const service = createService();
    await service.create({
      projectId: "project-a",
      name: "Staging",
      kind: "staging",
      slug: "staging",
    });

    await expect(
      service.create({
        projectId: "project-a",
        name: "Staging Copy",
        kind: "staging",
        slug: "staging",
      }),
    ).rejects.toBeInstanceOf(DuplicateEnvironmentSlugError);
  });

  it("lists environments immediately scoped by project", async () => {
    const service = createService();
    const created = await service.create({
      projectId: "project-a",
      name: "Development",
      kind: "development",
    });

    const listed = await service.list("project-a");
    expect(listed).toHaveLength(1);
    expect(listed[0]?.id).toBe(created.id);
  });

  it("defaults new environments to local_only", async () => {
    const service = createService();
    const environment = await service.create({
      projectId: "project-a",
      name: "Preview",
      kind: "preview",
    });

    expect(environment.status).toBe("local_only");
  });

  it("archives and restores environments", async () => {
    const service = createService();
    const created = await service.create({
      projectId: "project-a",
      name: "Production",
      kind: "production",
    });

    const archived = await service.archive(created.id);
    expect(archived.status).toBe("archived");

    const activeOnly = await service.list("project-a");
    expect(activeOnly).toHaveLength(0);

    const withArchived = await service.list("project-a", {
      includeArchived: true,
    });
    expect(withArchived).toHaveLength(1);

    const restored = await service.restore(created.id);
    expect(restored.status).toBe("local_only");
  });

  it("does not list environments from another project", async () => {
    const service = createService();
    await service.create({
      projectId: "project-a",
      name: "Local",
      kind: "local",
    });
    await service.create({
      projectId: "project-b",
      name: "Local",
      kind: "local",
    });

    const projectA = await service.list("project-a");
    const projectB = await service.list("project-b");

    expect(projectA).toHaveLength(1);
    expect(projectB).toHaveLength(1);
    expect(projectA[0]?.projectId).toBe("project-a");
    expect(projectB[0]?.projectId).toBe("project-b");
  });
});
