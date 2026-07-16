import { describe, expect, it } from "vitest";
import type {
  ApprovalRequestRecord,
  LocalClientPublicView,
  OperationRecord,
} from "@rayvan/daemon-contracts";

import {
  mapClientRows,
  mapOverview,
  mapSetupView,
} from "./mappers.js";

describe("agent-mcp mappers", () => {
  it("maps overview counts from clients, approvals, and daemon status", () => {
    const clients: LocalClientPublicView[] = [
      {
        id: "mcp-1",
        name: "Cursor",
        type: "mcp",
        status: "active",
        permissionProfileId: "operator",
        projectScopes: [],
        approvalPolicy: { type: "always_require_desktop_approval" },
        createdAt: "2026-01-01T00:00:00.000Z",
        connected: true,
      },
      {
        id: "rayvan-desktop",
        name: "Desktop",
        type: "desktop",
        status: "active",
        permissionProfileId: "administrator",
        projectScopes: [],
        approvalPolicy: {
          type: "client_may_approve",
          allowDestructive: false,
        },
        createdAt: "2026-01-01T00:00:00.000Z",
        connected: true,
      },
    ];
    const approvals: ApprovalRequestRecord[] = [
      {
        id: "appr-1",
        projectId: "proj-1",
        requestedBy: { type: "mcp_client", id: "mcp-1", displayName: "Cursor" },
        type: "remote_apply",
        summary: "Apply change",
        safeDetails: {},
        status: "pending",
        createdAt: "2026-01-02T00:00:00.000Z",
      },
    ];
    const operations: OperationRecord[] = [
      {
        id: "op-1",
        type: "findings_scan",
        status: "succeeded",
        actor: { type: "desktop", id: "rayvan-desktop" },
        correlationId: "c1",
      },
    ];

    const overview = mapOverview({
      clients,
      approvals,
      operations,
      snapshot: {
        connected: true,
        endpoint: "\\\\.\\pipe\\rayvan-test",
        spawned: false,
        daemonVersion: "0.0.1",
      },
      status: {
        state: "ready",
        version: "0.0.1",
        protocolVersion: "1",
        pid: 1,
        uptimeMs: 10,
        databasePath: "/tmp/rayvan.db",
        databaseSchemaVersion: 7,
        endpoint: "\\\\.\\pipe\\rayvan-test",
        connectedClients: 1,
        activeOperations: 0,
        pendingApprovals: 1,
        pluginHostStatus: "ready",
      },
    });

    expect(overview.mcpClientCount).toBe(1);
    expect(overview.activeMcpClientCount).toBe(1);
    expect(overview.pendingApprovalCount).toBe(1);
    expect(overview.recentOperationCount).toBe(1);
    expect(overview.daemonConnected).toBe(true);
    expect(overview.daemonState).toBe("ready");
  });

  it("maps client rows and setup snippet without exposing credentials", () => {
    const rows = mapClientRows([
      {
        id: "mcp-1",
        name: "Cursor",
        type: "mcp",
        status: "active",
        permissionProfileId: "operator",
        projectScopes: ["proj-1"],
        approvalPolicy: { type: "always_require_desktop_approval" },
        createdAt: "2026-01-01T00:00:00.000Z",
        connected: false,
      },
    ]);
    expect(rows[0]?.projectScopeLabel).toBe("proj-1");
    expect(rows[0]?.name).toBe("Cursor");

    const setup = mapSetupView("mcp-1");
    expect(setup.commandSnippet).toContain(
      "rayvan-mcp serve --client-id mcp-1",
    );
    expect(setup.commandSnippet.toLowerCase()).not.toContain("rvc_");
    expect(setup.notes[0]).toMatch(/keyring/i);
  });
});
