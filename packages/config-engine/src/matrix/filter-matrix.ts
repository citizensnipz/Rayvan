import type {
  ConfigurationMatrixCellStatus,
  ConfigurationMatrixRowViewModel,
  ConfigurationMatrixViewModel,
} from "./types.js";

export interface ConfigurationMatrixFilters {
  search?: string;
  statuses?: ConfigurationMatrixCellStatus[];
  requiredOnly?: boolean;
  sensitiveOnly?: boolean;
  /** Caller-resolved allow-list (e.g. from plugin occurrence lookup). */
  keyIdsAllowed?: Set<string>;
}

function recomputeSummary(
  matrix: ConfigurationMatrixViewModel,
): ConfigurationMatrixViewModel["summary"] {
  let missingCellCount = 0;
  let mismatchedCellCount = 0;
  let lockedCellCount = 0;
  let healthyCellCount = 0;

  for (const row of matrix.rows) {
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
    keyCount: matrix.rows.length,
    environmentCount: matrix.columns.length,
    missingCellCount,
    mismatchedCellCount,
    lockedCellCount,
    healthyCellCount,
  };
}

function rowMatchesFilters(
  row: ConfigurationMatrixRowViewModel,
  filters: ConfigurationMatrixFilters,
): boolean {
  if (filters.requiredOnly && !row.required) {
    return false;
  }
  if (filters.sensitiveOnly && !row.sensitive) {
    return false;
  }
  if (
    filters.keyIdsAllowed &&
    !filters.keyIdsAllowed.has(row.configurationKeyId)
  ) {
    return false;
  }
  if (filters.search) {
    const needle = filters.search.trim().toLowerCase();
    if (needle.length > 0 && !row.name.toLowerCase().includes(needle)) {
      return false;
    }
  }
  if (filters.statuses && filters.statuses.length > 0) {
    const allowed = new Set(filters.statuses);
    if (!row.cells.some((cell) => allowed.has(cell.status))) {
      return false;
    }
  }
  return true;
}

export function filterConfigurationMatrix(
  matrix: ConfigurationMatrixViewModel,
  filters: ConfigurationMatrixFilters,
): ConfigurationMatrixViewModel {
  const rows = matrix.rows.filter((row) => rowMatchesFilters(row, filters));
  const filtered: ConfigurationMatrixViewModel = {
    projectId: matrix.projectId,
    columns: matrix.columns,
    rows,
    summary: {
      keyCount: 0,
      environmentCount: matrix.columns.length,
      missingCellCount: 0,
      mismatchedCellCount: 0,
      lockedCellCount: 0,
      healthyCellCount: 0,
    },
  };
  filtered.summary = recomputeSummary(filtered);
  return filtered;
}
