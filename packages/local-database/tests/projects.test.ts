import { describe, expect, it } from "vitest";

import {
  InvalidProjectNameError,
  ProjectNotFoundError,
} from "../src/projects/errors.js";
import type { ProjectRepository } from "../src/projects/repository.js";
import { InMemoryProjectRepository } from "../src/repositories/index.js";

function createRepository(): ProjectRepository {
  return new InMemoryProjectRepository();
}

describe("ProjectRepository", () => {
  it("creates a project", async () => {
    const repository = createRepository();
    const project = await repository.create({
      name: "Payments API",
      description: "Core billing service",
    });

    expect(project.name).toBe("Payments API");
    expect(project.description).toBe("Core billing service");
    expect(project.status).toBe("active");
    expect(project.id).toBeTruthy();
    expect(project.createdAt).toBeTruthy();
    expect(project.updatedAt).toBeTruthy();
  });

  it("rejects an empty name", async () => {
    const repository = createRepository();
    await expect(repository.create({ name: "   " })).rejects.toBeInstanceOf(
      InvalidProjectNameError,
    );
  });

  it("lists projects with active-only default", async () => {
    const repository = createRepository();
    const active = await repository.create({ name: "Active" });
    const archived = await repository.create({ name: "Archived" });
    await repository.setArchived(archived.id, true);

    const activeOnly = await repository.list();
    expect(activeOnly).toHaveLength(1);
    expect(activeOnly[0]?.id).toBe(active.id);

    const withArchived = await repository.list({ includeArchived: true });
    expect(withArchived).toHaveLength(2);
  });

  it("loads a project by id", async () => {
    const repository = createRepository();
    const created = await repository.create({ name: "Lookup" });
    const loaded = await repository.getById(created.id);

    expect(loaded).toEqual(created);
    expect(await repository.getById("missing")).toBeNull();
  });

  it("edits a project", async () => {
    const repository = createRepository();
    const created = await repository.create({ name: "Original" });
    const updated = await repository.update(created.id, {
      name: "Renamed",
      description: "Updated description",
    });

    expect(updated.name).toBe("Renamed");
    expect(updated.description).toBe("Updated description");
    expect(updated.updatedAt >= created.updatedAt).toBe(true);
  });

  it("archives and restores a project", async () => {
    const repository = createRepository();
    const created = await repository.create({ name: "Toggle" });
    const archived = await repository.setArchived(created.id, true);
    expect(archived.status).toBe("archived");

    const restored = await repository.setArchived(created.id, false);
    expect(restored.status).toBe("active");
  });

  it("permanently deletes a project", async () => {
    const repository = createRepository();
    const created = await repository.create({ name: "Delete me" });
    await repository.delete(created.id);

    expect(await repository.getById(created.id)).toBeNull();
    await expect(repository.delete(created.id)).rejects.toBeInstanceOf(
      ProjectNotFoundError,
    );
  });
});
