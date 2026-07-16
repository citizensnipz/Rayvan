import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  FINDING_SCHEMA_VERSION,
  environmentId,
  findingEvaluationRunId,
  findingId,
  projectId,
  type FindingRecord,
} from "@rayvan/core";
import Database from "better-sqlite3";
import { describe, expect, it } from "vitest";

import { LocalDatabaseConnection } from "../src/database/connection.js";
import { MIGRATION_VERSION } from "../src/database/migrations.js";
import {
  createInMemoryFindingsPersistence,
  FindingLifecycleService,
  FindingSummaryService,
  InMemoryFindingLifecycleEventRepository,
  InMemoryFindingRepository,
  SqliteFindingEvaluationRunRepository,
  SqliteFindingLifecycleEventRepository,
  SqliteFindingRepository,
} from "../src/findings/index.js";
import { SqliteProjectRepository } from "../src/projects/sqlite-repository.js";

const NOW = "2026-07-16T12:00:00.000Z";
const ACTOR = { kind: "user" as const, id: "user-1", displayName: "Ada" };

function makeRecord(
  overrides: Partial<FindingRecord> & {
    fingerprint: string;
    ruleId?: string;
  },
): FindingRecord {
  return {
    id: findingId(overrides.id ?? crypto.randomUUID()),
    projectId: projectId(overrides.projectId ?? "project-a"),
    ruleId: overrides.ruleId ?? "cfg.missing-required",
    source: overrides.source ?? { type: "rayvan" },
    category: overrides.category ?? "configuration",
    severity: overrides.severity ?? "warning",
    title: overrides.title ?? "Missing required key",
    summary: overrides.summary ?? "DATABASE_URL is required",
    description: overrides.description,
    status: overrides.status ?? "open",
    fingerprint: overrides.fingerprint,
    fingerprintVersion: overrides.fingerprintVersion ?? "1",
    environmentId: overrides.environmentId,
    connectionId: overrides.connectionId,
    resourceBindingId: overrides.resourceBindingId,
    configurationKeyId: overrides.configurationKeyId,
    evidence: overrides.evidence ?? [{ type: "message", message: "missing" }],
    remediation: overrides.remediation,
    firstDetectedAt: overrides.firstDetectedAt ?? NOW,
    lastDetectedAt: overrides.lastDetectedAt ?? NOW,
    occurrenceCount: overrides.occurrenceCount ?? 1,
    acknowledgedAt: overrides.acknowledgedAt,
    acknowledgedBy: overrides.acknowledgedBy,
    dismissedAt: overrides.dismissedAt,
    dismissedBy: overrides.dismissedBy,
    dismissalReason: overrides.dismissalReason,
    resolvedAt: overrides.resolvedAt,
    resolution: overrides.resolution,
    suppressedUntil: overrides.suppressedUntil,
    lastEvaluationRunId:
      overrides.lastEvaluationRunId ?? findingEvaluationRunId("run-1"),
    metadata: overrides.metadata ?? {},
    schemaVersion: overrides.schemaVersion ?? FINDING_SCHEMA_VERSION,
  };
}

describe("FindingLifecycleService (memory)", () => {
  it("creates, saves, and gets by fingerprint", async () => {
    const db = createInMemoryFindingsPersistence();
    const record = makeRecord({ fingerprint: "fp-1" });
    await db.findings.save(record);

    const byId = await db.findings.getById(record.id);
    const byFingerprint = await db.findings.getByFingerprint(
      "project-a",
      "fp-1",
    );

    expect(byId?.id).toBe(record.id);
    expect(byFingerprint?.fingerprint).toBe("fp-1");
  });

  it("lists with filters", async () => {
    const db = createInMemoryFindingsPersistence();
    await db.findings.save(
      makeRecord({
        fingerprint: "fp-open",
        status: "open",
        severity: "error",
        environmentId: environmentId("env-dev"),
      }),
    );
    await db.findings.save(
      makeRecord({
        fingerprint: "fp-ack",
        status: "acknowledged",
        severity: "warning",
        connectionId: "conn-1",
      }),
    );
    await db.findings.save(
      makeRecord({
        fingerprint: "fp-resolved",
        status: "resolved",
        severity: "info",
      }),
    );

    const active = await db.service.list({ projectId: "project-a" });
    expect(active).toHaveLength(2);

    const errors = await db.service.list({
      projectId: "project-a",
      severities: ["error"],
      includeResolved: true,
    });
    expect(errors).toHaveLength(1);
    expect(errors[0]?.fingerprint).toBe("fp-open");

    const byEnv = await db.service.list({
      projectId: "project-a",
      environmentId: "env-dev",
    });
    expect(byEnv).toHaveLength(1);
  });

  it("acknowledge does not resolve", async () => {
    const db = createInMemoryFindingsPersistence();
    const record = makeRecord({ fingerprint: "fp-ack" });
    await db.findings.save(record);

    const acknowledged = await db.service.acknowledge(
      record.id,
      ACTOR,
      "looking into it",
      NOW,
    );

    expect(acknowledged.status).toBe("acknowledged");
    expect(acknowledged.resolvedAt).toBeUndefined();
    expect(acknowledged.acknowledgedBy).toEqual(ACTOR);

    const events = await db.lifecycleEvents.listByFindingId(record.id);
    expect(events).toHaveLength(1);
    expect(events[0]?.type).toBe("acknowledged");
    expect(events[0]?.reason).toBe("looking into it");
    expect(events[0]?.nextStatus).toBe("acknowledged");
  });

  it("dismisses a finding", async () => {
    const db = createInMemoryFindingsPersistence();
    const record = makeRecord({ fingerprint: "fp-dismiss" });
    await db.findings.save(record);

    const dismissed = await db.service.dismiss(
      record.id,
      ACTOR,
      "not applicable",
      NOW,
    );

    expect(dismissed.status).toBe("dismissed");
    expect(dismissed.dismissalReason).toBe("not applicable");
    expect(dismissed.dismissedBy).toEqual(ACTOR);

    const active = await db.service.list({ projectId: "project-a" });
    expect(active).toHaveLength(0);
  });

  it("suppresses with until", async () => {
    const db = createInMemoryFindingsPersistence();
    const record = makeRecord({ fingerprint: "fp-suppress" });
    await db.findings.save(record);
    const until = "2026-07-23T12:00:00.000Z";

    const suppressed = await db.service.suppress(
      record.id,
      ACTOR,
      { until, reason: "noise" },
      NOW,
    );

    expect(suppressed.status).toBe("suppressed");
    expect(suppressed.suppressedUntil).toBe(until);
  });

  it("suppresses with preset", async () => {
    const db = createInMemoryFindingsPersistence();
    const record = makeRecord({ fingerprint: "fp-suppress-preset" });
    await db.findings.save(record);

    const suppressed = await db.service.suppress(
      record.id,
      ACTOR,
      { preset: "24h" },
      NOW,
    );

    expect(suppressed.status).toBe("suppressed");
    expect(suppressed.suppressedUntil).toBe("2026-07-17T12:00:00.000Z");
  });

  it("lifecycle events are append-only", async () => {
    const findings = new InMemoryFindingRepository();
    const lifecycleEvents = new InMemoryFindingLifecycleEventRepository();
    const service = new FindingLifecycleService(findings, lifecycleEvents);
    const record = makeRecord({ fingerprint: "fp-life" });
    await findings.save(record);

    await service.acknowledge(record.id, ACTOR, undefined, NOW);
    await service.dismiss(record.id, ACTOR, "done", "2026-07-16T13:00:00.000Z");

    const events = await lifecycleEvents.listByFindingId(record.id);
    expect(events).toHaveLength(2);
    expect(events.map((event) => event.type)).toEqual([
      "acknowledged",
      "dismissed",
    ]);
    expect(lifecycleEvents.events).toHaveLength(2);
  });

  it("summary counts active findings", async () => {
    const db = createInMemoryFindingsPersistence();
    await db.findings.save(
      makeRecord({
        fingerprint: "fp-1",
        status: "open",
        severity: "error",
        remediation: { type: "manual", label: "Fix", instructions: "Set key" },
      }),
    );
    await db.findings.save(
      makeRecord({
        fingerprint: "fp-2",
        status: "acknowledged",
        severity: "warning",
      }),
    );
    await db.findings.save(
      makeRecord({
        fingerprint: "fp-3",
        status: "suppressed",
        severity: "info",
        connectionId: "conn-1",
        resourceBindingId: "binding-1",
        environmentId: environmentId("env-dev"),
      }),
    );
    await db.findings.save(
      makeRecord({
        fingerprint: "fp-4",
        status: "resolved",
        severity: "critical",
      }),
    );

    const summary = await db.summaryService.getProjectSummary("project-a");
    expect(summary.openCount).toBe(1);
    expect(summary.acknowledgedCount).toBe(1);
    expect(summary.bySeverity.error).toBe(1);
    expect(summary.bySeverity.warning).toBe(1);
    expect(summary.bySeverity.info).toBe(1);
    expect(summary.bySeverity.critical).toBe(0);
    expect(summary.hasRemediableFindings).toBe(true);
    expect(summary.highestSeverity).toBe("error");

    const envSummary = await db.summaryService.getEnvironmentSummary(
      "project-a",
      "env-dev",
    );
    expect(envSummary.openCount).toBe(0);
    expect(envSummary.bySeverity.info).toBe(1);

    const integrationSummary = await db.summaryService.getIntegrationSummary(
      "project-a",
      "conn-1",
    );
    expect(integrationSummary.bySeverity.info).toBe(1);

    const resourceSummary = await db.summaryService.getResourceSummary(
      "project-a",
      "binding-1",
    );
    expect(resourceSummary.bySeverity.info).toBe(1);
  });
});

function canUseBetterSqlite3(): boolean {
  try {
    const database = new Database(":memory:");
    database.close();
    return true;
  } catch {
    return false;
  }
}

const describeSqlite = canUseBetterSqlite3() ? describe : describe.skip;

describeSqlite("Finding sqlite repository", () => {
  it("applies v6 and round-trips findings", async () => {
    const directory = mkdtempSync(join(tmpdir(), "rayvan-findings-db-"));
    const dbPath = join(directory, "rayvan.db");

    try {
      const connection = new LocalDatabaseConnection(dbPath);
      const version = connection.raw
        .prepare("SELECT MAX(version) AS version FROM schema_migrations")
        .get() as { version: number };
      expect(version.version).toBe(MIGRATION_VERSION);

      const tables = connection.raw
        .prepare(
          "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'finding%'",
        )
        .all() as Array<{ name: string }>;
      expect(tables.map((row) => row.name).sort()).toEqual([
        "finding_evaluation_runs",
        "finding_lifecycle_events",
        "findings",
      ]);

      const projects = new SqliteProjectRepository(connection);
      await projects.create({
        name: "Findings Project",
        description: "sqlite round-trip",
      });
      const project = (await projects.list())[0];
      expect(project).toBeDefined();

      const findings = new SqliteFindingRepository(connection);
      const lifecycle = new SqliteFindingLifecycleEventRepository(connection);
      const evaluationRuns = new SqliteFindingEvaluationRunRepository(
        connection,
      );
      const service = new FindingLifecycleService(findings, lifecycle);
      const summaryService = new FindingSummaryService(findings);

      const record = makeRecord({
        id: findingId("finding-1"),
        projectId: project!.id,
        fingerprint: "fp-sqlite-1",
        title: "SQLite finding",
      });
      await findings.save(record);

      const loaded = await findings.getByFingerprint(project!.id, "fp-sqlite-1");
      expect(loaded?.title).toBe("SQLite finding");
      expect(loaded?.evidence).toEqual(record.evidence);

      const acknowledged = await service.acknowledge(
        record.id,
        ACTOR,
        "triaging",
        NOW,
      );
      expect(acknowledged.status).toBe("acknowledged");

      const events = await lifecycle.listByFindingId(record.id);
      expect(events).toHaveLength(1);
      expect(events[0]?.type).toBe("acknowledged");

      await evaluationRuns.save({
        id: findingEvaluationRunId("run-sqlite-1"),
        projectId: project!.id,
        scope: { type: "project", projectId: project!.id },
        trigger: "manual",
        status: "succeeded",
        startedAt: NOW,
        finishedAt: NOW,
        evaluatorsRun: 1,
        evaluatorsFailed: 0,
        findingsCreated: 1,
        findingsUpdated: 0,
        findingsReopened: 0,
        findingsResolved: 0,
        safeErrors: [],
      });
      const runs = await evaluationRuns.listByProject(project!.id);
      expect(runs).toHaveLength(1);

      const summary = await summaryService.getProjectSummary(project!.id);
      expect(summary.acknowledgedCount).toBe(1);

      connection.close();
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });
});
