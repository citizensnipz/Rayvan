import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { DaemonMethods } from "@rayvan/daemon-contracts";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { parseCliArguments } from "../src/cli.js";
import { createRayvanMcpServer } from "../src/server/index.js";
import { RAYVAN_PROMPT_NAMES } from "../src/prompts/index.js";
import { RAYVAN_TOOL_NAMES } from "../src/tools/index.js";

class FakeDaemonClient {
  calls: Array<{ method: string; params: unknown }> = [];
  error: (Error & { code?: string; data?: unknown }) | undefined;

  async call<T>(method: string, params?: unknown): Promise<T> {
    this.calls.push({ method, params });
    if (this.error) throw this.error;
    if (method === DaemonMethods.listProjects) {
      return [{ id: "project-1", name: "Demo" }] as T;
    }
    if (method === DaemonMethods.getProjectOverview) {
      return {
        project: { id: "project-1" },
        environmentCount: 2,
        openFindings: 1,
      } as T;
    }
    if (method === DaemonMethods.setConfigurationValue) {
      return { saved: { revision: 2 }, remoteStateAffected: false } as T;
    }
    return { ok: true } as T;
  }
}

describe("@rayvan/mcp-server", () => {
  let daemon: FakeDaemonClient;
  let client: Client;
  let server: ReturnType<typeof createRayvanMcpServer>;

  beforeEach(async () => {
    daemon = new FakeDaemonClient();
    server = createRayvanMcpServer(daemon);
    client = new Client({ name: "test", version: "1.0.0" });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);
    await client.connect(clientTransport);
  });

  afterEach(async () => {
    await client.close();
    await server.close();
  });

  it("accepts the serve client id flag or environment variable", () => {
    expect(parseCliArguments(["serve", "--client-id", "client-flag"], {})).toEqual({
      clientId: "client-flag",
    });
    expect(
      parseCliArguments(["serve"], { RAYVAN_MCP_CLIENT_ID: "client-env" }),
    ).toEqual({
      clientId: "client-env",
    });
    expect(() => parseCliArguments(["serve"], {})).toThrow("MCP client id is required");
    expect(() => parseCliArguments(["serve", "--credential", "secret"], {})).toThrow(
      "Unknown argument",
    );
  });

  it("lists the complete non-placeholder tool surface with risk annotations", async () => {
    const result = await client.listTools();
    const names = result.tools.map((tool) => tool.name);
    expect(names).toEqual(expect.arrayContaining(RAYVAN_TOOL_NAMES));
    expect(names).toContain("apply_change_plan");
    expect(
      result.tools.find((tool) => tool.name === "list_projects")?.annotations
        ?.readOnlyHint,
    ).toBe(true);
    expect(
      result.tools.find((tool) => tool.name === "apply_change_plan")?.annotations
        ?.destructiveHint,
    ).toBe(true);
    expect(JSON.stringify(result)).not.toContain("not implemented yet");
    expect(JSON.stringify(result)).not.toContain("Placeholder");
  });

  it("lists project resources and all six prompt templates", async () => {
    const resources = await client.listResources();
    const templates = await client.listResourceTemplates();
    const prompts = await client.listPrompts();
    expect(resources.resources.map((resource) => resource.uri)).toContain(
      "rayvan://projects",
    );
    expect(templates.resourceTemplates.map((template) => template.uriTemplate)).toEqual(
      expect.arrayContaining([
        "rayvan://projects/{projectId}",
        "rayvan://projects/{projectId}/environments",
      ]),
    );
    expect(prompts.prompts.map((prompt) => prompt.name)).toEqual(
      expect.arrayContaining([...RAYVAN_PROMPT_NAMES]),
    );
    expect(prompts.prompts).toHaveLength(6);
  });

  it("maps read tools and resources to daemon methods", async () => {
    const toolResult = await client.callTool({
      name: "get_project",
      arguments: { project_id: "project-1" },
    });
    expect(daemon.calls.at(-1)).toEqual({
      method: DaemonMethods.getProject,
      params: { projectId: "project-1" },
    });
    expect(toolResult.structuredContent).toEqual({ data: { ok: true } });

    const resource = await client.readResource({ uri: "rayvan://projects/project-1" });
    expect(daemon.calls.at(-1)).toEqual({
      method: DaemonMethods.getProjectOverview,
      params: { projectId: "project-1" },
    });
    expect(resource.contents[0]).toMatchObject({ mimeType: "application/json" });
  });

  it("maps local writes and returns structured daemon output", async () => {
    const result = await client.callTool({
      name: "set_configuration_value",
      arguments: {
        project_id: "project-1",
        environment_id: "env-1",
        configuration_key_id: "key-1",
        value: "false",
        expected_revision: 1,
      },
    });
    expect(daemon.calls.at(-1)).toEqual({
      method: DaemonMethods.setConfigurationValue,
      params: {
        projectId: "project-1",
        environmentId: "env-1",
        configurationKeyId: "key-1",
        value: "false",
        expectedRevision: 1,
      },
    });
    expect(result.isError).not.toBe(true);
    expect(result.structuredContent).toEqual({
      data: { saved: { revision: 2 }, remoteStateAffected: false },
    });
  });

  it("maps daemon failures to safe MCP errors without claiming success", async () => {
    const error = new Error("Project scope denied") as Error & {
      code?: string;
      data?: unknown;
    };
    error.code = "PROJECT_SCOPE_DENIED";
    error.data = {
      code: "PROJECT_SCOPE_DENIED",
      message: "Project scope denied",
      retryable: false,
      correlationId: "corr-safe",
      details: { secret: "must-not-leak" },
    };
    daemon.error = error;

    const result = await client.callTool({
      name: "get_project",
      arguments: { project_id: "outside-scope" },
    });
    expect(result.isError).toBe(true);
    expect(result.structuredContent).toEqual({
      data: {
        error: {
          code: "PROJECT_SCOPE_DENIED",
          message: "Project scope denied",
          retryable: false,
          correlationId: "corr-safe",
        },
      },
    });
    expect(JSON.stringify(result)).not.toContain("must-not-leak");
    expect(JSON.stringify(result)).not.toContain("completed through rayvand");
  });
});
