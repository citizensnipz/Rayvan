import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  V3_PLUGIN_PERSISTENCE_SQL,
  V4_ENVIRONMENTS_AND_CONFIGURATION_SQL,
  V5_DESIRED_APPLIED_CONFIGURATION_SQL,
  V6_FINDINGS_SQL,
  V7_DAEMON_CONTROL_PLANE_SQL,
} from "../src/database/migrations.js";

describe("V3 plugin persistence SQL parity", () => {
  it("matches the shared Rust migration SQL file byte-for-byte", () => {
    const here = dirname(fileURLToPath(import.meta.url));
    const rustSqlPath = join(
      here,
      "../../../crates/local-store/migrations/v3_plugin_persistence.sql",
    );
    const rustSql = readFileSync(rustSqlPath, "utf8").replace(/\r\n/g, "\n");
    expect(V3_PLUGIN_PERSISTENCE_SQL.replace(/\r\n/g, "\n")).toBe(rustSql);
  });

  it("Rust migrations.rs includes the shared SQL file", () => {
    const here = dirname(fileURLToPath(import.meta.url));
    const rustMigrations = readFileSync(
      join(here, "../../../crates/local-store/src/migrations.rs"),
      "utf8",
    );
    expect(rustMigrations).toContain(
      'include_str!("../migrations/v3_plugin_persistence.sql")',
    );
    expect(rustMigrations).toContain("CURRENT_SCHEMA_VERSION: u32 = 7");
  });
});

describe("V4 environments and configuration SQL parity", () => {
  it("matches the shared Rust migration SQL file byte-for-byte", () => {
    const here = dirname(fileURLToPath(import.meta.url));
    const rustSqlPath = join(
      here,
      "../../../crates/local-store/migrations/v4_environments_configuration.sql",
    );
    const rustSql = readFileSync(rustSqlPath, "utf8").replace(/\r\n/g, "\n");
    expect(V4_ENVIRONMENTS_AND_CONFIGURATION_SQL.replace(/\r\n/g, "\n")).toBe(rustSql);
  });

  it("Rust migrations.rs includes the shared SQL file", () => {
    const here = dirname(fileURLToPath(import.meta.url));
    const rustMigrations = readFileSync(
      join(here, "../../../crates/local-store/src/migrations.rs"),
      "utf8",
    );
    expect(rustMigrations).toContain(
      'include_str!("../migrations/v4_environments_configuration.sql")',
    );
  });
});

describe("V5 desired and applied configuration SQL parity", () => {
  it("matches the shared Rust migration SQL file byte-for-byte", () => {
    const here = dirname(fileURLToPath(import.meta.url));
    const rustSqlPath = join(
      here,
      "../../../crates/local-store/migrations/v5_desired_applied_configuration.sql",
    );
    const rustSql = readFileSync(rustSqlPath, "utf8").replace(/\r\n/g, "\n");
    expect(V5_DESIRED_APPLIED_CONFIGURATION_SQL.replace(/\r\n/g, "\n")).toBe(rustSql);
  });

  it("Rust migrations.rs includes the shared SQL file", () => {
    const here = dirname(fileURLToPath(import.meta.url));
    const rustMigrations = readFileSync(
      join(here, "../../../crates/local-store/src/migrations.rs"),
      "utf8",
    );
    expect(rustMigrations).toContain(
      'include_str!("../migrations/v5_desired_applied_configuration.sql")',
    );
  });
});

describe("V6 findings SQL parity", () => {
  it("matches the shared Rust migration SQL file byte-for-byte", () => {
    const here = dirname(fileURLToPath(import.meta.url));
    const rustSqlPath = join(
      here,
      "../../../crates/local-store/migrations/v6_findings.sql",
    );
    const rustSql = readFileSync(rustSqlPath, "utf8").replace(/\r\n/g, "\n");
    expect(V6_FINDINGS_SQL.replace(/\r\n/g, "\n")).toBe(rustSql);
  });

  it("Rust migrations.rs includes the shared SQL file", () => {
    const here = dirname(fileURLToPath(import.meta.url));
    const rustMigrations = readFileSync(
      join(here, "../../../crates/local-store/src/migrations.rs"),
      "utf8",
    );
    expect(rustMigrations).toContain('include_str!("../migrations/v6_findings.sql")');
  });

  it("defines findings tables and required indexes", () => {
    expect(V6_FINDINGS_SQL).toContain("CREATE TABLE IF NOT EXISTS findings");
    expect(V6_FINDINGS_SQL).toContain("finding_lifecycle_events");
    expect(V6_FINDINGS_SQL).toContain("finding_evaluation_runs");
    expect(V6_FINDINGS_SQL).toContain("UNIQUE (project_id, fingerprint)");
    expect(V6_FINDINGS_SQL).toContain("idx_findings_project_status");
    expect(V6_FINDINGS_SQL).toContain("idx_finding_lifecycle_events_finding_created");
    expect(V6_FINDINGS_SQL).toContain("idx_finding_evaluation_runs_project_started");
  });
});

describe("V7 daemon control plane SQL parity", () => {
  it("matches the shared Rust migration SQL file byte-for-byte", () => {
    const here = dirname(fileURLToPath(import.meta.url));
    const rustSqlPath = join(
      here,
      "../../../crates/local-store/migrations/v7_daemon_control_plane.sql",
    );
    const rustSql = readFileSync(rustSqlPath, "utf8").replace(/\r\n/g, "\n");
    expect(V7_DAEMON_CONTROL_PLANE_SQL.replace(/\r\n/g, "\n")).toBe(rustSql);
  });

  it("Rust migrations.rs includes the shared SQL file", () => {
    const here = dirname(fileURLToPath(import.meta.url));
    const rustMigrations = readFileSync(
      join(here, "../../../crates/local-store/src/migrations.rs"),
      "utf8",
    );
    expect(rustMigrations).toContain(
      'include_str!("../migrations/v7_daemon_control_plane.sql")',
    );
    expect(rustMigrations).toContain("CURRENT_SCHEMA_VERSION: u32 = 7");
  });

  it("defines daemon control-plane tables", () => {
    expect(V7_DAEMON_CONTROL_PLANE_SQL).toContain("local_clients");
    expect(V7_DAEMON_CONTROL_PLANE_SQL).toContain("operations");
    expect(V7_DAEMON_CONTROL_PLANE_SQL).toContain("approval_requests");
    expect(V7_DAEMON_CONTROL_PLANE_SQL).toContain("mcp_audit_events");
    expect(V7_DAEMON_CONTROL_PLANE_SQL).toContain("plugin_mcp_actions");
  });
});
