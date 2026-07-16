import type {
  FindingLifecycleEventRecord,
  FindingRecord,
  FindingSeverity,
  FindingSummary,
} from "@rayvan/core";

import {
  CATEGORY_LABELS,
  SEVERITY_LABELS,
  SEVERITY_ORDER,
  STATUS_LABELS,
  type FindingDetailViewModel,
  type FindingFiltersState,
  type FindingListItemViewModel,
  type FindingSeverityGroupViewModel,
} from "./view-models.js";

const PRODUCTION_HINT = /production/i;

function formatAbsolute(iso: string): string {
  try {
    return new Date(iso).toISOString();
  } catch {
    return iso;
  }
}

function formatLocal(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function sourceLabel(record: FindingRecord): string {
  if (record.source.type === "plugin") {
    return `Plugin · ${record.source.pluginId}${
      record.metadata.mockPluginFinding ? " (mock)" : ""
    }`;
  }
  return "Rayvan";
}

function isProductionFinding(record: FindingRecord): boolean {
  if (record.metadata.environmentKind === "production") {
    return true;
  }
  const envHint = String(record.environmentId ?? "");
  return PRODUCTION_HINT.test(envHint) || PRODUCTION_HINT.test(record.title);
}

export function mapFindingToListItem(
  record: FindingRecord,
  options?: {
    environmentNames?: Record<string, string>;
    connectionNames?: Record<string, string>;
  },
): FindingListItemViewModel {
  const environmentLabel = record.environmentId
    ? options?.environmentNames?.[record.environmentId] ??
      guessEnvironmentLabel(record.environmentId)
    : undefined;
  const integrationLabel = record.connectionId
    ? options?.connectionNames?.[record.connectionId] ??
      guessConnectionLabel(record)
    : undefined;

  return {
    findingId: record.id,
    title: record.title,
    summary: record.summary,
    severity: record.severity,
    severityLabel: SEVERITY_LABELS[record.severity],
    status: record.status,
    statusLabel: STATUS_LABELS[record.status],
    category: record.category,
    categoryLabel: CATEGORY_LABELS[record.category],
    environmentId: record.environmentId,
    environmentLabel,
    connectionId: record.connectionId,
    integrationLabel,
    sourceLabel: sourceLabel(record),
    lastDetectedAt: record.lastDetectedAt,
    lastDetectedLabel: formatLocal(record.lastDetectedAt),
    remediable: Boolean(record.remediation),
    isProduction: isProductionFinding(record) || PRODUCTION_HINT.test(environmentLabel ?? ""),
  };
}

function guessEnvironmentLabel(id: string): string {
  const match = /env:([^:]+)$/i.exec(id) ?? /:([^:]+)$/.exec(id);
  if (!match?.[1]) {
    return id;
  }
  return match[1]
    .split("-")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function guessConnectionLabel(record: FindingRecord): string {
  if (record.source.type === "plugin") {
    return record.source.pluginId;
  }
  return record.connectionId ?? "Integration";
}

export function filterFindings(
  records: FindingRecord[],
  filters: FindingFiltersState,
): FindingRecord[] {
  const search = filters.search.trim().toLowerCase();
  return records.filter((record) => {
    if (filters.openOnly) {
      if (record.status !== "open" && record.status !== "acknowledged") {
        return false;
      }
    } else if (filters.statuses.length > 0) {
      if (!filters.statuses.includes(record.status)) {
        return false;
      }
    }

    if (
      filters.severities.length > 0 &&
      !filters.severities.includes(record.severity)
    ) {
      return false;
    }
    if (
      filters.categories.length > 0 &&
      !filters.categories.includes(record.category)
    ) {
      return false;
    }
    if (
      filters.environmentId &&
      record.environmentId !== filters.environmentId
    ) {
      return false;
    }
    if (filters.connectionId && record.connectionId !== filters.connectionId) {
      return false;
    }
    if (filters.remediableOnly && !record.remediation) {
      return false;
    }
    if (search) {
      const haystack = [
        record.title,
        record.summary,
        record.description ?? "",
        record.ruleId,
        record.category,
        record.severity,
      ]
        .join(" ")
        .toLowerCase();
      if (!haystack.includes(search)) {
        return false;
      }
    }
    return true;
  });
}

export function groupFindingsBySeverity(
  items: FindingListItemViewModel[],
): FindingSeverityGroupViewModel[] {
  const bySeverity = new Map<FindingSeverity, FindingListItemViewModel[]>();
  for (const severity of SEVERITY_ORDER) {
    bySeverity.set(severity, []);
  }
  for (const item of items) {
    bySeverity.get(item.severity)?.push(item);
  }

  for (const [, group] of bySeverity) {
    group.sort((a, b) => {
      if (a.isProduction !== b.isProduction) {
        return a.isProduction ? -1 : 1;
      }
      return b.lastDetectedAt.localeCompare(a.lastDetectedAt);
    });
  }

  return SEVERITY_ORDER.map((severity) => ({
    severity,
    severityLabel: SEVERITY_LABELS[severity],
    items: bySeverity.get(severity) ?? [],
  })).filter((group) => group.items.length > 0);
}

export function mapFindingDetail(
  record: FindingRecord,
  lifecycleEvents: FindingLifecycleEventRecord[],
  options?: {
    environmentNames?: Record<string, string>;
    connectionNames?: Record<string, string>;
  },
): FindingDetailViewModel {
  return {
    finding: record,
    severityLabel: SEVERITY_LABELS[record.severity],
    statusLabel: STATUS_LABELS[record.status],
    categoryLabel: CATEGORY_LABELS[record.category],
    sourceLabel: sourceLabel(record),
    environmentLabel: record.environmentId
      ? options?.environmentNames?.[record.environmentId] ??
        guessEnvironmentLabel(record.environmentId)
      : undefined,
    integrationLabel: record.connectionId
      ? options?.connectionNames?.[record.connectionId] ??
        guessConnectionLabel(record)
      : undefined,
    firstDetectedLabel: formatLocal(record.firstDetectedAt),
    lastDetectedLabel: formatLocal(record.lastDetectedAt),
    firstDetectedAbsolute: formatAbsolute(record.firstDetectedAt),
    lastDetectedAbsolute: formatAbsolute(record.lastDetectedAt),
    remediation: record.remediation,
    lifecycleEvents: [...lifecycleEvents].sort((a, b) =>
      b.createdAt.localeCompare(a.createdAt),
    ),
    canAcknowledge: record.status === "open" || record.status === "acknowledged",
    canDismiss:
      record.status === "open" ||
      record.status === "acknowledged" ||
      record.status === "suppressed",
    canSuppress:
      record.status === "open" ||
      record.status === "acknowledged" ||
      record.status === "suppressed",
  };
}

export function openFindingsCountFromSummary(summary: FindingSummary): number {
  return summary.openCount + summary.acknowledgedCount;
}
