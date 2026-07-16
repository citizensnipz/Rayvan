import type { FindingSummary } from "@rayvan/core";
import {
  summarizeFindings,
  type FindingRepository,
} from "@rayvan/findings-engine";

/**
 * Aggregates FindingSummary values from persisted findings.
 * Lists ACTIVE statuses only, then uses findings-engine `summarizeFindings`.
 */
export class FindingSummaryService {
  constructor(private readonly findings: FindingRepository) {}

  async getProjectSummary(projectId: string): Promise<FindingSummary> {
    const records = await this.findings.list({
      projectId,
      includeResolved: false,
    });
    return summarizeFindings(records);
  }

  async getEnvironmentSummary(
    projectId: string,
    environmentId: string,
  ): Promise<FindingSummary> {
    const records = await this.findings.list({
      projectId,
      environmentId,
      includeResolved: false,
    });
    return summarizeFindings(records);
  }

  async getIntegrationSummary(
    projectId: string,
    connectionId: string,
  ): Promise<FindingSummary> {
    const records = await this.findings.list({
      projectId,
      connectionId,
      includeResolved: false,
    });
    return summarizeFindings(records);
  }

  async getResourceSummary(
    projectId: string,
    resourceBindingId: string,
  ): Promise<FindingSummary> {
    const records = await this.findings.list({
      projectId,
      resourceBindingId,
      includeResolved: false,
    });
    return summarizeFindings(records);
  }
}
