import { describe, expect, it } from "vitest";

import {
  ConfigurationDesiredStateService,
  DesiredConfigurationRevisionConflictError,
  InMemoryAppliedConfigurationStateRepository,
  InMemoryConfigurationKeyRepository,
  InMemoryDesiredConfigurationValueRepository,
  PlaintextSecretNotAllowedError,
} from "../src/configuration/index.js";

function createService(): {
  keys: InMemoryConfigurationKeyRepository;
  service: ConfigurationDesiredStateService;
} {
  const keys = new InMemoryConfigurationKeyRepository();
  const desired = new InMemoryDesiredConfigurationValueRepository();
  const applied = new InMemoryAppliedConfigurationStateRepository();
  return {
    keys,
    service: new ConfigurationDesiredStateService(keys, desired, applied),
  };
}

const ACTOR = { kind: "user" as const, id: "user-1", displayName: "Test" };

describe("ConfigurationDesiredStateService", () => {
  it("saves desired value for a key × environment", async () => {
    const { keys, service } = createService();
    const key = await keys.create({
      projectId: "project-a",
      name: "API_BASE_URL",
      valueType: "url",
      required: true,
      sensitive: false,
      source: "manual",
    });

    const saved = await service.saveDesired({
      configurationKeyId: key.id,
      environmentId: "env-prod",
      projectId: "project-a",
      desiredValue: "https://api.example.com",
      valueFingerprint: "fp:api",
      updatedBy: ACTOR,
    });

    expect(saved.desiredValue).toBe("https://api.example.com");
    expect(saved.revision).toBe(1);
    expect(saved.environmentId).toBe("env-prod");

    const listed = await service.listByEnvironment("env-prod");
    expect(listed).toHaveLength(1);
  });

  it("rejects concurrency conflicts on expectedRevision", async () => {
    const { keys, service } = createService();
    const key = await keys.create({
      projectId: "project-a",
      name: "NODE_ENV",
      valueType: "string",
      required: true,
      sensitive: false,
      source: "manual",
    });

    await service.saveDesired({
      configurationKeyId: key.id,
      environmentId: "env-prod",
      projectId: "project-a",
      desiredValue: "production",
      updatedBy: ACTOR,
    });

    await service.saveDesired({
      configurationKeyId: key.id,
      environmentId: "env-prod",
      projectId: "project-a",
      desiredValue: "staging",
      expectedRevision: 1,
      updatedBy: ACTOR,
    });

    await expect(
      service.saveDesired({
        configurationKeyId: key.id,
        environmentId: "env-prod",
        projectId: "project-a",
        desiredValue: "development",
        expectedRevision: 1,
        updatedBy: ACTOR,
      }),
    ).rejects.toBeInstanceOf(DesiredConfigurationRevisionConflictError);
  });

  it("rejects plaintext secrets in desiredValue", async () => {
    const { keys, service } = createService();
    const key = await keys.create({
      projectId: "project-a",
      name: "API_SECRET",
      valueType: "secret",
      required: true,
      sensitive: true,
      source: "manual",
    });

    await expect(
      service.saveDesired({
        configurationKeyId: key.id,
        environmentId: "env-prod",
        projectId: "project-a",
        desiredValue: "sk_live_plaintext",
        updatedBy: ACTOR,
      }),
    ).rejects.toBeInstanceOf(PlaintextSecretNotAllowedError);

    const saved = await service.saveDesired({
      configurationKeyId: key.id,
      environmentId: "env-prod",
      projectId: "project-a",
      secretValueRef: "cred:api-secret",
      valueFingerprint: "fp:secret",
      updatedBy: ACTOR,
    });

    expect(saved.desiredValue).toBeUndefined();
    expect(saved.secretValueRef).toBe("cred:api-secret");
  });

  it("records applied state per resource binding", async () => {
    const { keys, service } = createService();
    const key = await keys.create({
      projectId: "project-a",
      name: "API_BASE_URL",
      valueType: "url",
      required: true,
      sensitive: false,
      source: "manual",
    });

    const desired = await service.saveDesired({
      configurationKeyId: key.id,
      environmentId: "env-prod",
      projectId: "project-a",
      desiredValue: "https://api.example.com",
      valueFingerprint: "fp:api",
      updatedBy: ACTOR,
    });

    const applied = await service.recordApplied({
      configurationKeyId: key.id,
      environmentId: "env-prod",
      projectId: "project-a",
      resourceBindingId: "binding-1",
      desiredRevision: desired.revision,
      appliedFingerprint: "fp:api",
      applyExecutionId: "exec-1",
      status: "applied",
    });

    expect(applied.resourceBindingId).toBe("binding-1");
    expect(applied.desiredRevision).toBe(1);
    expect(applied.status).toBe("applied");

    const listed = await service.listAppliedByEnvironment("env-prod");
    expect(listed).toHaveLength(1);
  });
});
