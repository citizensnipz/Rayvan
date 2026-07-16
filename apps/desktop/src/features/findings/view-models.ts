import type {
  FindingCategory,
  FindingLifecycleEventRecord,
  FindingRecord,
  FindingRemediation,
  FindingSeverity,
  FindingStatus,
  FindingSummary,
} from "@rayvan/core";

export type FindingsWorkspaceTabKind = "list" | "detail";

export type FindingsTab =
  | { kind: "list" }
  | { kind: "detail"; findingId: string; label: string };

export type FindingEmptyVariant =
  | "no-project"
  | "never-evaluated"
  | "no-active"
  | "scanning"
  | "partial-failure"
  | "no-matches";

export interface FindingFiltersState {
  search: string;
  statuses: FindingStatus[];
  severities: FindingSeverity[];
  categories: FindingCategory[];
  environmentId: string | null;
  connectionId: string | null;
  remediableOnly: boolean;
  /** When true, show open + acknowledged (default). */
  openOnly: boolean;
}

export interface FindingListItemViewModel {
  findingId: string;
  title: string;
  summary: string;
  severity: FindingSeverity;
  severityLabel: string;
  status: FindingStatus;
  statusLabel: string;
  category: FindingCategory;
  categoryLabel: string;
  environmentId?: string;
  environmentLabel?: string;
  connectionId?: string;
  integrationLabel?: string;
  sourceLabel: string;
  lastDetectedAt: string;
  lastDetectedLabel: string;
  remediable: boolean;
  isProduction: boolean;
}

export interface FindingSeverityGroupViewModel {
  severity: FindingSeverity;
  severityLabel: string;
  items: FindingListItemViewModel[];
}

export interface FindingHeaderSummaryViewModel {
  openCount: number;
  criticalCount: number;
  warningCount: number;
  informationalCount: number;
  highestSeverity?: FindingSeverity;
  headline: string;
}

export interface FindingDetailViewModel {
  finding: FindingRecord;
  severityLabel: string;
  statusLabel: string;
  categoryLabel: string;
  sourceLabel: string;
  environmentLabel?: string;
  integrationLabel?: string;
  firstDetectedLabel: string;
  lastDetectedLabel: string;
  firstDetectedAbsolute: string;
  lastDetectedAbsolute: string;
  remediation?: FindingRemediation;
  lifecycleEvents: FindingLifecycleEventRecord[];
  canAcknowledge: boolean;
  canDismiss: boolean;
  canSuppress: boolean;
}

export interface FindingEnvironmentOption {
  id: string;
  label: string;
}

export interface FindingIntegrationOption {
  id: string;
  label: string;
}

export const SEVERITY_ORDER: readonly FindingSeverity[] = [
  "critical",
  "error",
  "warning",
  "info",
] as const;

export const SEVERITY_LABELS: Record<FindingSeverity, string> = {
  critical: "Critical",
  error: "Error",
  warning: "Warning",
  info: "Informational",
};

export const STATUS_LABELS: Record<FindingStatus, string> = {
  open: "Open",
  acknowledged: "Acknowledged",
  resolved: "Resolved",
  dismissed: "Dismissed",
  suppressed: "Suppressed",
};

export const CATEGORY_LABELS: Record<FindingCategory, string> = {
  configuration: "Configuration",
  environment: "Environment",
  integration: "Integration",
  resource: "Resource",
  deployment: "Deployment",
  security: "Security",
  availability: "Availability",
  drift: "Drift",
  permission: "Permission",
  mapping: "Mapping",
  other: "Other",
};

export function defaultFindingFilters(): FindingFiltersState {
  return {
    search: "",
    statuses: [],
    severities: [],
    categories: [],
    environmentId: null,
    connectionId: null,
    remediableOnly: false,
    openOnly: true,
  };
}

export function mapHeaderSummary(
  summary: FindingSummary | null,
): FindingHeaderSummaryViewModel {
  if (!summary) {
    return {
      openCount: 0,
      criticalCount: 0,
      warningCount: 0,
      informationalCount: 0,
      headline: "Findings",
    };
  }
  const openCount = summary.openCount + summary.acknowledgedCount;
  const criticalCount = summary.bySeverity.critical;
  const warningCount = summary.bySeverity.warning + summary.bySeverity.error;
  const informationalCount = summary.bySeverity.info;
  const parts = [
    `${openCount} open`,
    `${criticalCount} critical`,
    `${warningCount} warning${warningCount === 1 ? "" : "s"}`,
    `${informationalCount} informational`,
  ];
  return {
    openCount,
    criticalCount,
    warningCount,
    informationalCount,
    highestSeverity: summary.highestSeverity,
    headline: parts.join(" · "),
  };
}
