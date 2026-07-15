import { describe, expect, it } from "vitest";

import {
  ConfigurationService,
  InMemoryConfigurationKeyRepository,
  InMemoryConfigurationOccurrenceRepository,
  PlaintextSecretNotAllowedError,
} from "../src/configuration/index.js";

function createService(): ConfigurationService {
  return new ConfigurationService(
    new InMemoryConfigurationKeyRepository(),
    new InMemoryConfigurationOccurrenceRepository(),
  );
}

describe("ConfigurationService", () => {
  it("upserts a key by name", async () => {
    const service = createService();
    const created = await service.upsertKeyByName("project-a", "DATABASE_URL", {
      valueType: "url",
      required: true,
    });

    expect(created.name).toBe("DATABASE_URL");
    expect(created.valueType).toBe("url");
    expect(created.required).toBe(true);
    expect(created.sensitive).toBe(false);
    expect(created.source).toBe("manual");

    const updated = await service.upsertKeyByName("project-a", "DATABASE_URL", {
      description: "Primary database",
      required: false,
    });

    expect(updated.id).toBe(created.id);
    expect(updated.description).toBe("Primary database");
    expect(updated.required).toBe(false);

    const keys = await service.listKeys("project-a");
    expect(keys).toHaveLength(1);
  });

  it("allows one logical key with multiple occurrences", async () => {
    const service = createService();
    const key = await service.upsertKeyByName("project-a", "API_URL", {
      valueType: "url",
    });

    const first = await service.upsertOccurrence({
      configurationKeyId: key.id,
      projectId: "project-a",
      environmentId: "env-local",
      pluginId: "vercel",
      connectionId: "conn-1",
      discoveredResourceId: "res-1",
      providerKey: "API_URL",
      valueAccess: "readable",
      observedValue: "https://local.example",
    });

    const second = await service.upsertOccurrence({
      configurationKeyId: key.id,
      projectId: "project-a",
      environmentId: "env-prod",
      pluginId: "vercel",
      connectionId: "conn-1",
      discoveredResourceId: "res-2",
      providerKey: "API_URL",
      valueAccess: "readable",
      observedValue: "https://prod.example",
    });

    expect(first.id).not.toBe(second.id);

    const byKey = await service.listOccurrencesByKey(key.id);
    expect(byKey).toHaveLength(2);

    const byProject = await service.listOccurrencesByProject("project-a");
    expect(byProject).toHaveLength(2);

    const byEnv = await service.listOccurrencesByEnvironment("env-local");
    expect(byEnv).toHaveLength(1);
    expect(byEnv[0]?.observedValue).toBe("https://local.example");
  });

  it("rejects plaintext secrets in observedValue when sensitive", async () => {
    const service = createService();
    const key = await service.upsertKeyByName("project-a", "API_SECRET", {
      valueType: "secret",
      sensitive: true,
    });

    await expect(
      service.upsertOccurrence({
        configurationKeyId: key.id,
        projectId: "project-a",
        pluginId: "vercel",
        connectionId: "conn-1",
        discoveredResourceId: "res-1",
        providerKey: "API_SECRET",
        valueAccess: "readable",
        observedValue: "super-secret-value",
      }),
    ).rejects.toBeInstanceOf(PlaintextSecretNotAllowedError);

    const stored = await service.upsertOccurrence({
      configurationKeyId: key.id,
      projectId: "project-a",
      pluginId: "vercel",
      connectionId: "conn-1",
      discoveredResourceId: "res-1",
      providerKey: "API_SECRET",
      valueAccess: "readable",
      secretValueRef: "cred:api-secret",
      maskedValue: "••••",
      valueFingerprint: "fp-1",
    });

    expect(stored.observedValue).toBeUndefined();
    expect(stored.secretValueRef).toBe("cred:api-secret");
  });
});
