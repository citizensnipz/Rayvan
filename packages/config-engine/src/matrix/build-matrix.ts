import type {
  ConfigurationKey,
  ConfigurationOccurrence,
  Environment,
} from "@rayvan/core";
import type {
  ConfigurationDerivedFinding,
  ConfigurationMatrixCellStatus,
  ConfigurationMatrixCellViewModel,
  ConfigurationMatrixColumnViewModel,
  ConfigurationMatrixRowViewModel,
  ConfigurationMatrixViewModel,
} from "./types.js";

const OPAQUE_ACCESS = new Set(["locked", "name_only", "masked"]);

const STATUS_LABELS: Record<ConfigurationMatrixCellStatus, string> = {
  healthy: "Healthy",
  present: "Present",
  missing: "Missing",
  mismatched: "Different",
  partially_present: "Partially present",
  locked: "Value locked",
  unknown: "Unknown",
  not_applicable: "Not applicable",
};

function statusLabel(status: ConfigurationMatrixCellStatus): string {
  return STATUS_LABELS[status];
}

function isComparableReadable(
  key: ConfigurationKey,
  occurrence: ConfigurationOccurrence,
): boolean {
  if (occurrence.valueAccess !== "readable") {
    return false;
  }
  if (key.sensitive) {
    return Boolean(occurrence.valueFingerprint);
  }
  return (
    occurrence.valueFingerprint !== undefined ||
    occurrence.observedValue !== undefined
  );
}

/** Prefer fingerprints when every occurrence has one; otherwise use observed values. */
function agreementTokens(
  key: ConfigurationKey,
  occurrences: ConfigurationOccurrence[],
): Array<string | undefined> {
  const allHaveFingerprints = occurrences.every(
    (occurrence) => occurrence.valueFingerprint !== undefined,
  );
  if (allHaveFingerprints) {
    return occurrences.map(
      (occurrence) => `fp:${occurrence.valueFingerprint as string}`,
    );
  }
  if (key.sensitive) {
    // Sensitive values are only comparable via fingerprint.
    return occurrences.map((occurrence) =>
      occurrence.valueFingerprint
        ? `fp:${occurrence.valueFingerprint}`
        : undefined,
    );
  }
  return occurrences.map((occurrence) =>
    occurrence.observedValue !== undefined
      ? `val:${occurrence.observedValue}`
      : undefined,
  );
}

function buildCell(
  key: ConfigurationKey,
  environmentId: string,
  cellOccurrences: ConfigurationOccurrence[],
  keyHasAnyMappedOccurrence: boolean,
): ConfigurationMatrixCellViewModel {
  const occurrenceIds = cellOccurrences.map((occurrence) => occurrence.id);
  const accessLocked = cellOccurrences.some(
    (occurrence) => occurrence.valueAccess === "locked",
  );

  if (cellOccurrences.length === 0) {
    const status: ConfigurationMatrixCellStatus = key.required
      ? "missing"
      : keyHasAnyMappedOccurrence
        ? "missing"
        : "not_applicable";
    const requiredMissing = key.required;
    return {
      configurationKeyId: key.id,
      configurationKeyName: key.name,
      environmentId,
      status,
      statusLabel: statusLabel(status),
      occurrenceCount: 0,
      accessLocked: false,
      requiredMissing,
      warningCount: requiredMissing ? 1 : 0,
      occurrenceIds: [],
    };
  }

  const missingAccess = cellOccurrences.filter(
    (occurrence) => occurrence.valueAccess === "missing",
  );
  const presentAccess = cellOccurrences.filter(
    (occurrence) => occurrence.valueAccess !== "missing",
  );
  const requiredMissing =
    key.required &&
    (missingAccess.length > 0 || presentAccess.length === 0);

  if (presentAccess.length === 0) {
    const status: ConfigurationMatrixCellStatus = "missing";
    return {
      configurationKeyId: key.id,
      configurationKeyName: key.name,
      environmentId,
      status,
      statusLabel: statusLabel(status),
      occurrenceCount: cellOccurrences.length,
      accessLocked: false,
      requiredMissing,
      warningCount: requiredMissing ? 1 : 0,
      occurrenceIds,
    };
  }

  if (missingAccess.length > 0 && presentAccess.length > 0) {
    const status: ConfigurationMatrixCellStatus = "partially_present";
    return {
      configurationKeyId: key.id,
      configurationKeyName: key.name,
      environmentId,
      status,
      statusLabel: statusLabel(status),
      occurrenceCount: cellOccurrences.length,
      accessLocked,
      requiredMissing,
      warningCount: requiredMissing ? 1 : 0,
      occurrenceIds,
    };
  }

  const readable = presentAccess.filter(
    (occurrence) => occurrence.valueAccess === "readable",
  );
  const opaque = presentAccess.filter((occurrence) =>
    OPAQUE_ACCESS.has(occurrence.valueAccess),
  );

  // Present but not comparable (locked / name_only / masked only).
  if (readable.length === 0 && opaque.length > 0) {
    const status: ConfigurationMatrixCellStatus = opaque.some(
      (occurrence) => occurrence.valueAccess === "locked",
    )
      ? "locked"
      : "present";
    return {
      configurationKeyId: key.id,
      configurationKeyName: key.name,
      environmentId,
      status,
      statusLabel: statusLabel(status),
      occurrenceCount: cellOccurrences.length,
      // Never infer agreement among opaque access modes.
      valuesAgree: undefined,
      accessLocked: accessLocked || status === "locked",
      requiredMissing,
      warningCount:
        (requiredMissing ? 1 : 0) +
        (opaque.length >= 2 &&
        opaque.every(
          (occurrence) =>
            occurrence.valueAccess === "locked" ||
            occurrence.valueAccess === "name_only",
        )
          ? 1
          : 0),
      occurrenceIds,
    };
  }

  // Mix of readable and opaque — never treat opaque as agreeing with readable.
  if (readable.length > 0 && opaque.length > 0) {
    const comparable = readable.filter((occurrence) =>
      isComparableReadable(key, occurrence),
    );
    if (comparable.length >= 2) {
      const tokens = agreementTokens(key, comparable);
      if (
        tokens.every((token) => token !== undefined) &&
        new Set(tokens).size > 1
      ) {
        const status: ConfigurationMatrixCellStatus = "mismatched";
        return {
          configurationKeyId: key.id,
          configurationKeyName: key.name,
          environmentId,
          status,
          statusLabel: statusLabel(status),
          occurrenceCount: cellOccurrences.length,
          valuesAgree: false,
          accessLocked,
          requiredMissing,
          warningCount: 1 + (requiredMissing ? 1 : 0),
          occurrenceIds,
        };
      }
    }

    const status: ConfigurationMatrixCellStatus = "unknown";
    return {
      configurationKeyId: key.id,
      configurationKeyName: key.name,
      environmentId,
      status,
      statusLabel: statusLabel(status),
      occurrenceCount: cellOccurrences.length,
      valuesAgree: undefined,
      accessLocked,
      requiredMissing,
      warningCount: 1 + (requiredMissing ? 1 : 0),
      occurrenceIds,
    };
  }

  // All present occurrences are readable.
  const comparable = readable.filter((occurrence) =>
    isComparableReadable(key, occurrence),
  );

  if (comparable.length !== readable.length || comparable.length === 0) {
    const status: ConfigurationMatrixCellStatus = "present";
    return {
      configurationKeyId: key.id,
      configurationKeyName: key.name,
      environmentId,
      status,
      statusLabel: statusLabel(status),
      occurrenceCount: cellOccurrences.length,
      valuesAgree: undefined,
      accessLocked,
      requiredMissing,
      warningCount: requiredMissing ? 1 : 0,
      occurrenceIds,
    };
  }

  const tokens = agreementTokens(key, comparable);
  if (tokens.some((token) => token === undefined)) {
    const status: ConfigurationMatrixCellStatus = "unknown";
    return {
      configurationKeyId: key.id,
      configurationKeyName: key.name,
      environmentId,
      status,
      statusLabel: statusLabel(status),
      occurrenceCount: cellOccurrences.length,
      valuesAgree: undefined,
      accessLocked,
      requiredMissing,
      warningCount: 1 + (requiredMissing ? 1 : 0),
      occurrenceIds,
    };
  }

  const uniqueTokens = new Set(tokens);
  if (uniqueTokens.size > 1) {
    const status: ConfigurationMatrixCellStatus = "mismatched";
    return {
      configurationKeyId: key.id,
      configurationKeyName: key.name,
      environmentId,
      status,
      statusLabel: statusLabel(status),
      occurrenceCount: cellOccurrences.length,
      valuesAgree: false,
      accessLocked,
      requiredMissing,
      warningCount: 1 + (requiredMissing ? 1 : 0),
      occurrenceIds,
    };
  }

  const status: ConfigurationMatrixCellStatus = "healthy";
  let safeVisibleValue: string | undefined;
  if (!key.sensitive) {
    const observedValues = comparable
      .map((occurrence) => occurrence.observedValue)
      .filter((value): value is string => value !== undefined);
    if (
      observedValues.length === comparable.length &&
      new Set(observedValues).size === 1
    ) {
      safeVisibleValue = observedValues[0];
    }
  }

  return {
    configurationKeyId: key.id,
    configurationKeyName: key.name,
    environmentId,
    status,
    statusLabel: statusLabel(status),
    occurrenceCount: cellOccurrences.length,
    valuesAgree: true,
    accessLocked,
    requiredMissing,
    safeVisibleValue,
    warningCount: requiredMissing ? 1 : 0,
    occurrenceIds,
  };
}

function summarize(
  columns: ConfigurationMatrixColumnViewModel[],
  rows: ConfigurationMatrixRowViewModel[],
): ConfigurationMatrixViewModel["summary"] {
  let missingCellCount = 0;
  let mismatchedCellCount = 0;
  let lockedCellCount = 0;
  let healthyCellCount = 0;

  for (const row of rows) {
    for (const cell of row.cells) {
      switch (cell.status) {
        case "missing":
          missingCellCount += 1;
          break;
        case "mismatched":
          mismatchedCellCount += 1;
          break;
        case "locked":
          lockedCellCount += 1;
          break;
        case "healthy":
          healthyCellCount += 1;
          break;
        default:
          break;
      }
    }
  }

  return {
    keyCount: rows.length,
    environmentCount: columns.length,
    missingCellCount,
    mismatchedCellCount,
    lockedCellCount,
    healthyCellCount,
  };
}

export function buildConfigurationMatrix(input: {
  projectId: string;
  environments: Environment[];
  keys: ConfigurationKey[];
  occurrences: ConfigurationOccurrence[];
}): ConfigurationMatrixViewModel {
  const { projectId, environments, keys, occurrences } = input;

  const columns: ConfigurationMatrixColumnViewModel[] = environments.map(
    (environment) => ({
      environmentId: environment.id,
      name: environment.name,
      slug: environment.slug,
      kind: environment.kind,
    }),
  );

  const mappedByKey = new Map<string, ConfigurationOccurrence[]>();
  for (const occurrence of occurrences) {
    if (!occurrence.environmentId) {
      continue;
    }
    const list = mappedByKey.get(occurrence.configurationKeyId) ?? [];
    list.push(occurrence);
    mappedByKey.set(occurrence.configurationKeyId, list);
  }

  const rows: ConfigurationMatrixRowViewModel[] = keys.map((key) => {
    const keyOccurrences = mappedByKey.get(key.id) ?? [];
    const keyHasAnyMappedOccurrence = keyOccurrences.length > 0;

    const cells = environments.map((environment) => {
      const cellOccurrences = keyOccurrences.filter(
        (occurrence) => occurrence.environmentId === environment.id,
      );
      return buildCell(
        key,
        environment.id,
        cellOccurrences,
        keyHasAnyMappedOccurrence,
      );
    });

    return {
      configurationKeyId: key.id,
      name: key.name,
      required: key.required,
      sensitive: key.sensitive,
      valueType: key.valueType,
      cells,
    };
  });

  return {
    projectId,
    columns,
    rows,
    summary: summarize(columns, rows),
  };
}

export function buildConfigurationDerivedFindings(
  matrix: ConfigurationMatrixViewModel,
  keys: ConfigurationKey[],
  occurrences: ConfigurationOccurrence[],
): ConfigurationDerivedFinding[] {
  const findings: ConfigurationDerivedFinding[] = [];
  const keyById = new Map<string, ConfigurationKey>(
    keys.map((key) => [key.id, key]),
  );

  for (const row of matrix.rows) {
    for (const cell of row.cells) {
      if (cell.requiredMissing || cell.status === "missing") {
        const key = keyById.get(row.configurationKeyId);
        if (key?.required || cell.requiredMissing) {
          findings.push({
            id: `missing:${matrix.projectId}:${row.configurationKeyId}:${cell.environmentId}`,
            projectId: matrix.projectId,
            environmentId: cell.environmentId,
            severity: "error",
            category: "missing_configuration",
            title: "Missing required configuration",
            description: `${row.name} is required but missing in this environment.`,
            configurationKeyId: row.configurationKeyId,
            configurationKeyName: row.name,
          });
        }
      }

      if (cell.status === "mismatched") {
        findings.push({
          id: `drift:${matrix.projectId}:${row.configurationKeyId}:${cell.environmentId}`,
          projectId: matrix.projectId,
          environmentId: cell.environmentId,
          severity: "warning",
          category: "configuration_drift",
          title: "Configuration mismatch",
          description: `${row.name} has disagreeing values across sources in this environment.`,
          configurationKeyId: row.configurationKeyId,
          configurationKeyName: row.name,
        });
      }
    }
  }

  // Unknown value consistency: two+ locked/name_only for same key+env.
  const opaqueByCell = new Map<string, ConfigurationOccurrence[]>();
  for (const occurrence of occurrences) {
    if (!occurrence.environmentId) {
      continue;
    }
    if (
      occurrence.valueAccess !== "locked" &&
      occurrence.valueAccess !== "name_only"
    ) {
      continue;
    }
    const cellKey = `${occurrence.configurationKeyId}:${occurrence.environmentId}`;
    const list = opaqueByCell.get(cellKey) ?? [];
    list.push(occurrence);
    opaqueByCell.set(cellKey, list);
  }

  for (const [cellKey, opaqueOccurrences] of opaqueByCell) {
    if (opaqueOccurrences.length < 2) {
      continue;
    }
    const [configurationKeyId, environmentId] = cellKey.split(":");
    const key = keyById.get(configurationKeyId);
    findings.push({
      id: `unknown-consistency:${matrix.projectId}:${configurationKeyId}:${environmentId}`,
      projectId: matrix.projectId,
      environmentId,
      severity: "info",
      category: "health",
      title: "Unknown value consistency",
      description: `${key?.name ?? configurationKeyId} has multiple locked or name-only sources; values cannot be compared.`,
      configurationKeyId,
      configurationKeyName: key?.name,
    });
  }

  return findings;
}
