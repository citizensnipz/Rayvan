import type {
  FindingCategory,
  FindingRecord,
  FindingSeverity,
  FindingSummary,
} from "@rayvan/core";
import { FINDING_SEVERITIES } from "@rayvan/core";

const SEVERITY_RANK: Record<FindingSeverity, number> = {
  info: 0,
  warning: 1,
  error: 2,
  critical: 3,
};

/** Pure summary from FindingRecord[] — no I/O. */
export function summarizeFindings(records: FindingRecord[]): FindingSummary {
  const bySeverity: Record<FindingSeverity, number> = {
    info: 0,
    warning: 0,
    error: 0,
    critical: 0,
  };
  const byCategory: Partial<Record<FindingCategory, number>> = {};
  let openCount = 0;
  let acknowledgedCount = 0;
  let hasRemediableFindings = false;
  let highestSeverity: FindingSeverity | undefined;

  for (const record of records) {
    if (record.status === "open") {
      openCount += 1;
    }
    if (record.status === "acknowledged") {
      acknowledgedCount += 1;
    }
    if (
      record.status === "open" ||
      record.status === "acknowledged" ||
      record.status === "suppressed"
    ) {
      bySeverity[record.severity] += 1;
      byCategory[record.category] = (byCategory[record.category] ?? 0) + 1;
      if (record.remediation) {
        hasRemediableFindings = true;
      }
      if (
        highestSeverity === undefined ||
        SEVERITY_RANK[record.severity] > SEVERITY_RANK[highestSeverity]
      ) {
        highestSeverity = record.severity;
      }
    }
  }

  // Ensure stable key order for bySeverity
  for (const severity of FINDING_SEVERITIES) {
    bySeverity[severity] = bySeverity[severity] ?? 0;
  }

  return {
    openCount,
    acknowledgedCount,
    bySeverity,
    byCategory,
    highestSeverity,
    hasRemediableFindings,
  };
}
