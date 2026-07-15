import { describe, expect, it } from "vitest";
import {
  configurationKeyId,
  configurationOccurrenceId,
  environmentId,
  projectId,
  type ConfigurationKey,
  type ConfigurationOccurrence,
  type Environment,
} from "@rayvan/core";
import {
  buildConfigurationDerivedFindings,
  buildConfigurationMatrix,
  filterConfigurationMatrix,
} from "../src/matrix/index.js";

const PROJECT = projectId("project-1");
const ENV_DEV = environmentId("env-dev");
const ENV_PROD = environmentId("env-prod");
const NOW = "2026-07-15T00:00:00.000Z";

function makeEnvironment(
  id: ReturnType<typeof environmentId>,
  name: string,
  slug: string,
): Environment {
  return {
    id,
    projectId: PROJECT,
    name,
    slug,
    kind: slug === "prod" ? "production" : "development",
    status: "healthy",
    createdAt: NOW,
    updatedAt: NOW,
  };
}

function makeKey(
  overrides: Partial<ConfigurationKey> & Pick<ConfigurationKey, "id" | "name">,
): ConfigurationKey {
  return {
    projectId: PROJECT,
    description: undefined,
    valueType: "string",
    required: false,
    sensitive: false,
    source: "discovered",
    createdAt: NOW,
    updatedAt: NOW,
    ...overrides,
  };
}

function makeOccurrence(
  overrides: Partial<ConfigurationOccurrence> &
    Pick<
      ConfigurationOccurrence,
      "id" | "configurationKeyId" | "environmentId" | "valueAccess"
    >,
): ConfigurationOccurrence {
  return {
    projectId: PROJECT,
    pluginId: "vercel",
    connectionId: "conn-1",
    discoveredResourceId: "res-1",
    providerKey: "KEY",
    firstObservedAt: NOW,
    lastObservedAt: NOW,
    ...overrides,
  };
}

describe("buildConfigurationMatrix", () => {
  const environments = [
    makeEnvironment(ENV_DEV, "Development", "dev"),
    makeEnvironment(ENV_PROD, "Production", "prod"),
  ];

  it("builds correct rows and columns in input order", () => {
    const keys = [
      makeKey({ id: configurationKeyId("key-a"), name: "API_URL" }),
      makeKey({ id: configurationKeyId("key-b"), name: "FEATURE_FLAG" }),
    ];

    const matrix = buildConfigurationMatrix({
      projectId: PROJECT,
      environments,
      keys,
      occurrences: [],
    });

    expect(matrix.columns.map((column) => column.slug)).toEqual([
      "dev",
      "prod",
    ]);
    expect(matrix.rows.map((row) => row.name)).toEqual([
      "API_URL",
      "FEATURE_FLAG",
    ]);
    expect(matrix.rows[0]?.cells).toHaveLength(2);
    expect(matrix.summary.keyCount).toBe(2);
    expect(matrix.summary.environmentCount).toBe(2);
  });

  it("marks matching readable values as healthy with safeVisibleValue", () => {
    const key = makeKey({
      id: configurationKeyId("key-url"),
      name: "API_URL",
    });
    const occurrences = [
      makeOccurrence({
        id: configurationOccurrenceId("occ-1"),
        configurationKeyId: key.id,
        environmentId: ENV_DEV,
        valueAccess: "readable",
        observedValue: "https://api.dev",
        valueFingerprint: "fp-dev",
      }),
      makeOccurrence({
        id: configurationOccurrenceId("occ-2"),
        configurationKeyId: key.id,
        environmentId: ENV_DEV,
        valueAccess: "readable",
        observedValue: "https://api.dev",
        valueFingerprint: "fp-dev",
        pluginId: "railway",
        connectionId: "conn-2",
      }),
    ];

    const matrix = buildConfigurationMatrix({
      projectId: PROJECT,
      environments,
      keys: [key],
      occurrences,
    });

    const cell = matrix.rows[0]?.cells[0];
    expect(cell?.status).toBe("healthy");
    expect(cell?.statusLabel).toBe("Healthy");
    expect(cell?.valuesAgree).toBe(true);
    expect(cell?.occurrenceCount).toBe(2);
    expect(cell?.safeVisibleValue).toBe("https://api.dev");
    expect(matrix.summary.healthyCellCount).toBe(1);
  });

  it("marks mismatched readable values as mismatched", () => {
    const key = makeKey({
      id: configurationKeyId("key-url"),
      name: "API_URL",
    });
    const occurrences = [
      makeOccurrence({
        id: configurationOccurrenceId("occ-1"),
        configurationKeyId: key.id,
        environmentId: ENV_DEV,
        valueAccess: "readable",
        observedValue: "https://api.dev",
        valueFingerprint: "fp-a",
      }),
      makeOccurrence({
        id: configurationOccurrenceId("occ-2"),
        configurationKeyId: key.id,
        environmentId: ENV_DEV,
        valueAccess: "readable",
        observedValue: "https://api.other",
        valueFingerprint: "fp-b",
        pluginId: "railway",
      }),
    ];

    const matrix = buildConfigurationMatrix({
      projectId: PROJECT,
      environments,
      keys: [key],
      occurrences,
    });

    const cell = matrix.rows[0]?.cells[0];
    expect(cell?.status).toBe("mismatched");
    expect(cell?.statusLabel).toBe("Different");
    expect(cell?.valuesAgree).toBe(false);
    expect(matrix.summary.mismatchedCellCount).toBe(1);
  });

  it("treats locked values as locked/present and never infers a match", () => {
    const key = makeKey({
      id: configurationKeyId("key-secret"),
      name: "SECRET",
      sensitive: true,
      valueType: "secret",
    });
    const occurrences = [
      makeOccurrence({
        id: configurationOccurrenceId("occ-1"),
        configurationKeyId: key.id,
        environmentId: ENV_DEV,
        valueAccess: "locked",
      }),
      makeOccurrence({
        id: configurationOccurrenceId("occ-2"),
        configurationKeyId: key.id,
        environmentId: ENV_DEV,
        valueAccess: "locked",
        pluginId: "railway",
      }),
    ];

    const matrix = buildConfigurationMatrix({
      projectId: PROJECT,
      environments,
      keys: [key],
      occurrences,
    });

    const cell = matrix.rows[0]?.cells[0];
    expect(cell?.status).toBe("locked");
    expect(cell?.statusLabel).toBe("Value locked");
    expect(cell?.valuesAgree).toBeUndefined();
    expect(cell?.accessLocked).toBe(true);
    expect(cell?.safeVisibleValue).toBeUndefined();
  });

  it("treats name-only secrets as present without agreement", () => {
    const key = makeKey({
      id: configurationKeyId("key-secret"),
      name: "API_TOKEN",
      sensitive: true,
      valueType: "secret",
    });
    const occurrences = [
      makeOccurrence({
        id: configurationOccurrenceId("occ-1"),
        configurationKeyId: key.id,
        environmentId: ENV_DEV,
        valueAccess: "name_only",
      }),
    ];

    const matrix = buildConfigurationMatrix({
      projectId: PROJECT,
      environments,
      keys: [key],
      occurrences,
    });

    const cell = matrix.rows[0]?.cells[0];
    expect(cell?.status).toBe("present");
    expect(cell?.valuesAgree).toBeUndefined();
    expect(cell?.safeVisibleValue).toBeUndefined();
  });

  it("marks missing required keys", () => {
    const key = makeKey({
      id: configurationKeyId("key-req"),
      name: "DATABASE_URL",
      required: true,
    });

    const matrix = buildConfigurationMatrix({
      projectId: PROJECT,
      environments,
      keys: [key],
      occurrences: [],
    });

    for (const cell of matrix.rows[0]?.cells ?? []) {
      expect(cell.status).toBe("missing");
      expect(cell.requiredMissing).toBe(true);
      expect(cell.statusLabel).toBe("Missing");
    }
    expect(matrix.summary.missingCellCount).toBe(2);
  });

  it("uses not_applicable for optional keys with no project occurrences", () => {
    const key = makeKey({
      id: configurationKeyId("key-opt"),
      name: "OPTIONAL_FLAG",
      required: false,
    });

    const matrix = buildConfigurationMatrix({
      projectId: PROJECT,
      environments,
      keys: [key],
      occurrences: [],
    });

    expect(matrix.rows[0]?.cells.every((cell) => cell.status === "not_applicable")).toBe(
      true,
    );
  });

  it("never puts sensitive values in safeVisibleValue", () => {
    const key = makeKey({
      id: configurationKeyId("key-secret"),
      name: "SECRET_KEY",
      sensitive: true,
      valueType: "secret",
    });
    const occurrences = [
      makeOccurrence({
        id: configurationOccurrenceId("occ-1"),
        configurationKeyId: key.id,
        environmentId: ENV_DEV,
        valueAccess: "readable",
        observedValue: "super-secret",
        valueFingerprint: "fp-secret",
      }),
    ];

    const matrix = buildConfigurationMatrix({
      projectId: PROJECT,
      environments,
      keys: [key],
      occurrences,
    });

    const cell = matrix.rows[0]?.cells[0];
    expect(cell?.status).toBe("healthy");
    expect(cell?.safeVisibleValue).toBeUndefined();
  });

  it("counts multiple sources as occurrenceCount", () => {
    const key = makeKey({
      id: configurationKeyId("key-multi"),
      name: "SHARED",
    });
    const occurrences = [
      makeOccurrence({
        id: configurationOccurrenceId("occ-1"),
        configurationKeyId: key.id,
        environmentId: ENV_DEV,
        valueAccess: "readable",
        observedValue: "a",
        valueFingerprint: "fp-a",
      }),
      makeOccurrence({
        id: configurationOccurrenceId("occ-2"),
        configurationKeyId: key.id,
        environmentId: ENV_DEV,
        valueAccess: "readable",
        observedValue: "a",
        valueFingerprint: "fp-a",
        pluginId: "railway",
      }),
      makeOccurrence({
        id: configurationOccurrenceId("occ-3"),
        configurationKeyId: key.id,
        environmentId: ENV_DEV,
        valueAccess: "readable",
        observedValue: "a",
        valueFingerprint: "fp-a",
        pluginId: "supabase",
      }),
    ];

    const matrix = buildConfigurationMatrix({
      projectId: PROJECT,
      environments,
      keys: [key],
      occurrences,
    });

    expect(matrix.rows[0]?.cells[0]?.occurrenceCount).toBe(3);
    expect(matrix.rows[0]?.cells[0]?.occurrenceIds).toHaveLength(3);
  });

  it("excludes unmapped occurrences without environmentId from env cells", () => {
    const key = makeKey({
      id: configurationKeyId("key-unmap"),
      name: "ORPHAN",
    });
    const occurrences = [
      makeOccurrence({
        id: configurationOccurrenceId("occ-1"),
        configurationKeyId: key.id,
        environmentId: undefined,
        valueAccess: "readable",
        observedValue: "x",
      }),
    ];

    const matrix = buildConfigurationMatrix({
      projectId: PROJECT,
      environments,
      keys: [key],
      occurrences,
    });

    expect(matrix.rows[0]?.cells.every((cell) => cell.occurrenceCount === 0)).toBe(
      true,
    );
    expect(matrix.rows[0]?.cells[0]?.status).toBe("not_applicable");
  });
});

describe("filterConfigurationMatrix", () => {
  it("filters by search and requiredOnly", () => {
    const environments = [
      makeEnvironment(ENV_DEV, "Development", "dev"),
      makeEnvironment(ENV_PROD, "Production", "prod"),
    ];
    const keys = [
      makeKey({
        id: configurationKeyId("key-1"),
        name: "DATABASE_URL",
        required: true,
      }),
      makeKey({
        id: configurationKeyId("key-2"),
        name: "FEATURE_FLAG",
        required: false,
      }),
      makeKey({
        id: configurationKeyId("key-3"),
        name: "DEBUG_MODE",
        required: true,
      }),
    ];

    const matrix = buildConfigurationMatrix({
      projectId: PROJECT,
      environments,
      keys,
      occurrences: [],
    });

    const requiredOnly = filterConfigurationMatrix(matrix, {
      requiredOnly: true,
    });
    expect(requiredOnly.rows.map((row) => row.name)).toEqual([
      "DATABASE_URL",
      "DEBUG_MODE",
    ]);

    const searched = filterConfigurationMatrix(matrix, {
      search: "data",
    });
    expect(searched.rows.map((row) => row.name)).toEqual(["DATABASE_URL"]);
  });
});

describe("buildConfigurationDerivedFindings", () => {
  it("emits findings for missing required and mismatched cells", () => {
    const environments = [
      makeEnvironment(ENV_DEV, "Development", "dev"),
      makeEnvironment(ENV_PROD, "Production", "prod"),
    ];
    const requiredKey = makeKey({
      id: configurationKeyId("key-req"),
      name: "DATABASE_URL",
      required: true,
    });
    const driftKey = makeKey({
      id: configurationKeyId("key-drift"),
      name: "API_URL",
    });

    const occurrences = [
      makeOccurrence({
        id: configurationOccurrenceId("occ-1"),
        configurationKeyId: driftKey.id,
        environmentId: ENV_DEV,
        valueAccess: "readable",
        observedValue: "a",
        valueFingerprint: "fp-a",
      }),
      makeOccurrence({
        id: configurationOccurrenceId("occ-2"),
        configurationKeyId: driftKey.id,
        environmentId: ENV_DEV,
        valueAccess: "readable",
        observedValue: "b",
        valueFingerprint: "fp-b",
        pluginId: "railway",
      }),
      makeOccurrence({
        id: configurationOccurrenceId("occ-3"),
        configurationKeyId: driftKey.id,
        environmentId: ENV_DEV,
        valueAccess: "locked",
        pluginId: "supabase",
      }),
      makeOccurrence({
        id: configurationOccurrenceId("occ-4"),
        configurationKeyId: driftKey.id,
        environmentId: ENV_DEV,
        valueAccess: "name_only",
        pluginId: "github",
      }),
    ];

    // Separate key for unknown consistency (two locked/name_only only)
    const opaqueKey = makeKey({
      id: configurationKeyId("key-opaque"),
      name: "TOKEN",
      sensitive: true,
    });
    const opaqueOccurrences = [
      makeOccurrence({
        id: configurationOccurrenceId("occ-5"),
        configurationKeyId: opaqueKey.id,
        environmentId: ENV_PROD,
        valueAccess: "locked",
      }),
      makeOccurrence({
        id: configurationOccurrenceId("occ-6"),
        configurationKeyId: opaqueKey.id,
        environmentId: ENV_PROD,
        valueAccess: "name_only",
        pluginId: "railway",
      }),
    ];

    const allOccurrences = [...occurrences, ...opaqueOccurrences];
    const keys = [requiredKey, driftKey, opaqueKey];
    const matrix = buildConfigurationMatrix({
      projectId: PROJECT,
      environments,
      keys,
      occurrences: allOccurrences,
    });

    const findings = buildConfigurationDerivedFindings(
      matrix,
      keys,
      allOccurrences,
    );

    expect(
      findings.some(
        (finding) =>
          finding.category === "missing_configuration" &&
          finding.configurationKeyName === "DATABASE_URL",
      ),
    ).toBe(true);
    expect(
      findings.some(
        (finding) =>
          finding.category === "configuration_drift" &&
          finding.configurationKeyName === "API_URL",
      ),
    ).toBe(true);
    expect(
      findings.some(
        (finding) =>
          finding.title === "Unknown value consistency" &&
          finding.configurationKeyName === "TOKEN",
      ),
    ).toBe(true);
  });
});
