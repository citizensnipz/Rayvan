export type ConfigurationMatrixCellStatus =
  | "healthy"
  | "present"
  | "missing"
  | "mismatched"
  | "partially_present"
  | "locked"
  | "unknown"
  | "not_applicable";

export interface ConfigurationMatrixCellViewModel {
  configurationKeyId: string;
  configurationKeyName: string;
  environmentId: string;
  status: ConfigurationMatrixCellStatus;
  statusLabel: string;
  occurrenceCount: number;
  valuesAgree?: boolean;
  accessLocked: boolean;
  requiredMissing: boolean;
  /** Safe non-sensitive visible value only; never secrets */
  safeVisibleValue?: string;
  warningCount: number;
  occurrenceIds: string[];
}

export interface ConfigurationMatrixRowViewModel {
  configurationKeyId: string;
  name: string;
  required: boolean;
  sensitive: boolean;
  valueType: string;
  cells: ConfigurationMatrixCellViewModel[]; // one per environment column order
}

export interface ConfigurationMatrixColumnViewModel {
  environmentId: string;
  name: string;
  slug: string;
  kind: string;
}

export interface ConfigurationMatrixViewModel {
  projectId: string;
  columns: ConfigurationMatrixColumnViewModel[];
  rows: ConfigurationMatrixRowViewModel[];
  summary: {
    keyCount: number;
    environmentCount: number;
    missingCellCount: number;
    mismatchedCellCount: number;
    lockedCellCount: number;
    healthyCellCount: number;
  };
}

/**
 * @deprecated Prefer FindingDetection from `@rayvan/findings-engine`.
 * Categories use the product Findings taxonomy for transitional callers.
 */
export interface ConfigurationDerivedFinding {
  id: string;
  projectId: string;
  environmentId?: string;
  severity: "info" | "warning" | "error";
  /** Product taxonomy: configuration | drift | other (was missing_configuration / configuration_drift / health). */
  category: "configuration" | "drift" | "other";
  title: string;
  description: string;
  configurationKeyId?: string;
  configurationKeyName?: string;
}
