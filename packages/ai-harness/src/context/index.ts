import type { Finding } from "@rayvan/core";

export interface InfrastructureContext {
  projectId: string;
  findings: Finding[];
}

export function summarizeFindings(findings: Finding[]): string {
  if (findings.length === 0) {
    return "No findings detected.";
  }
  return `${findings.length} finding(s) require attention.`;
}
