import { describe, expect, it } from "vitest";

import { createDevPluginIntegrationsGateway } from "./dev-gateway.js";

describe("createDevPluginIntegrationsGateway", () => {
  it("seeds the canonical fixture connections and catalog for a project", async () => {
    const gateway = createDevPluginIntegrationsGateway();
    await gateway.ensureProjectSeeded("project-1");

    const installed = await gateway.listInstalledPlugins();
    expect(installed.map((plugin) => plugin.pluginId).sort()).toEqual(
      ["example-local", "github", "runpod", "sentry", "supabase", "vercel"].sort(),
    );

    const connections = await gateway.listConnectionsByProject("project-1");
    expect(connections).toHaveLength(5);
    expect(connections.some((connection) => connection.pluginId === "runpod")).toBe(false);
  });

  it("installs runpod as a non-built-in catalog entry with no connection", async () => {
    const gateway = createDevPluginIntegrationsGateway();
    await gateway.ensureProjectSeeded("project-1");

    const installed = await gateway.listInstalledPlugins();
    const runpod = installed.find((plugin) => plugin.pluginId === "runpod");
    expect(runpod).toBeDefined();
    expect(runpod?.source.type).toBe("package");
  });

  it("does not duplicate connections when seeded twice for the same project", async () => {
    const gateway = createDevPluginIntegrationsGateway();
    await gateway.ensureProjectSeeded("project-1");
    await gateway.ensureProjectSeeded("project-1");

    const connections = await gateway.listConnectionsByProject("project-1");
    expect(connections).toHaveLength(5);
  });

  it("seeds fixtures independently for each project id the first time it is seen", async () => {
    const gateway = createDevPluginIntegrationsGateway();
    await gateway.ensureProjectSeeded("project-1");
    await gateway.ensureProjectSeeded("project-2");

    const connectionsA = await gateway.listConnectionsByProject("project-1");
    const connectionsB = await gateway.listConnectionsByProject("project-2");
    expect(connectionsA).toHaveLength(5);
    expect(connectionsB).toHaveLength(5);
    expect(connectionsA[0]?.id).not.toBe(connectionsB[0]?.id);
  });

  it("grants active permissions for connections that request them", async () => {
    const gateway = createDevPluginIntegrationsGateway();
    await gateway.ensureProjectSeeded("project-1");

    const connections = await gateway.listConnectionsByProject("project-1");
    const vercelConnection = connections.find((connection) => connection.pluginId === "vercel");
    expect(vercelConnection).toBeDefined();

    const grants = await gateway.listPermissionGrants(vercelConnection?.id ?? "");
    expect(grants.length).toBeGreaterThan(0);
    expect(grants.every((grant) => grant.granted)).toBe(true);
  });

  it("creates a fresh isolated instance per call, with no shared singleton state", async () => {
    const gatewayA = createDevPluginIntegrationsGateway();
    const gatewayB = createDevPluginIntegrationsGateway();

    await gatewayA.ensureProjectSeeded("project-1");

    const connectionsA = await gatewayA.listConnectionsByProject("project-1");
    const connectionsB = await gatewayB.listConnectionsByProject("project-1");

    expect(connectionsA.length).toBeGreaterThan(0);
    expect(connectionsB).toHaveLength(0);
  });

  it("supports creating a connection and granting permissions directly", async () => {
    const gateway = createDevPluginIntegrationsGateway();
    await gateway.ensureProjectSeeded("project-1");

    const installed = await gateway.listInstalledPlugins();
    const runpod = installed.find((plugin) => plugin.pluginId === "runpod");
    expect(runpod).toBeDefined();

    const connection = await gateway.createConnection({
      installedPluginId: runpod?.id ?? "",
      projectId: "project-1",
      name: "My RunPod",
    });
    expect(connection.status).toBe("connected");
    expect(connection.projectId).toBe("project-1");

    const grants = await gateway.grantPermissions({
      pluginId: "runpod",
      connectionId: connection.id,
      projectId: "project-1",
      permissions: ["network"],
      grantedBy: { type: "user", id: "test-user" },
    });
    expect(grants).toHaveLength(1);
    expect(grants[0]?.granted).toBe(true);
  });
});
