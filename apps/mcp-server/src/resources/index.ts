import type { DaemonClient } from "@rayvan/daemon-client";
import { DaemonMethods } from "@rayvan/daemon-contracts";
import { McpServer, ResourceTemplate } from "@modelcontextprotocol/sdk/server/mcp.js";

type DaemonCaller = Pick<DaemonClient, "call">;

export const RAYVAN_RESOURCE_URIS = [
  "rayvan://projects",
  "rayvan://projects/{projectId}",
  "rayvan://projects/{projectId}/environments",
  "rayvan://projects/{projectId}/environments/{environmentId}",
  "rayvan://projects/{projectId}/integrations",
  "rayvan://projects/{projectId}/configuration",
  "rayvan://projects/{projectId}/findings",
  "rayvan://projects/{projectId}/operations",
] as const;

export function registerRayvanResources(server: McpServer, daemon: DaemonCaller): void {
  server.registerResource(
    "projects",
    "rayvan://projects",
    {
      title: "Rayvan projects",
      description: "Read-only collection of projects visible to this MCP client.",
      mimeType: "application/json",
    },
    async (uri) => resource(uri, await daemon.call(DaemonMethods.listProjects, {})),
  );

  server.registerResource(
    "project_snapshot",
    new ResourceTemplate("rayvan://projects/{projectId}", {
      list: async () => {
        const projects = await daemon.call<unknown[]>(DaemonMethods.listProjects, {});
        return {
          resources: projects.flatMap((project) => {
            const record = asRecord(project);
            return typeof record.id === "string"
              ? [
                  {
                    uri: `rayvan://projects/${encodeURIComponent(record.id)}`,
                    name: typeof record.name === "string" ? record.name : record.id,
                    mimeType: "application/json",
                  },
                ]
              : [];
          }),
        };
      },
    }),
    {
      title: "Rayvan project snapshot",
      description: "Read-only project overview snapshot.",
      mimeType: "application/json",
    },
    async (uri, variables) =>
      resource(
        uri,
        await daemon.call(DaemonMethods.getProjectOverview, {
          projectId: String(variables.projectId),
        }),
      ),
  );

  server.registerResource(
    "project_environments",
    new ResourceTemplate("rayvan://projects/{projectId}/environments", {
      list: undefined,
    }),
    {
      title: "Rayvan project environments",
      description: "Read-only environment collection snapshot for a project.",
      mimeType: "application/json",
    },
    async (uri, variables) =>
      resource(
        uri,
        await daemon.call(DaemonMethods.listEnvironments, {
          projectId: String(variables.projectId),
        }),
      ),
  );

  registerProjectCollection(
    server,
    daemon,
    "project_integrations",
    "rayvan://projects/{projectId}/integrations",
    "Rayvan project integrations",
    DaemonMethods.listIntegrations,
  );
  registerProjectCollection(
    server,
    daemon,
    "project_configuration",
    "rayvan://projects/{projectId}/configuration",
    "Rayvan project configuration keys",
    DaemonMethods.listConfigurationKeys,
  );
  registerProjectCollection(
    server,
    daemon,
    "project_findings",
    "rayvan://projects/{projectId}/findings",
    "Rayvan project Findings",
    DaemonMethods.listFindings,
  );
  registerProjectCollection(
    server,
    daemon,
    "project_operations",
    "rayvan://projects/{projectId}/operations",
    "Rayvan project operations",
    DaemonMethods.listOperations,
  );

  server.registerResource(
    "environment_snapshot",
    new ResourceTemplate("rayvan://projects/{projectId}/environments/{environmentId}", {
      list: undefined,
    }),
    {
      title: "Rayvan environment snapshot",
      description: "Read-only environment snapshot visible to this MCP client.",
      mimeType: "application/json",
    },
    async (uri, variables) =>
      resource(
        uri,
        await daemon.call(DaemonMethods.getEnvironment, {
          projectId: String(variables.projectId),
          environmentId: String(variables.environmentId),
        }),
      ),
  );
}

function registerProjectCollection(
  server: McpServer,
  daemon: DaemonCaller,
  name: string,
  uriTemplate: string,
  title: string,
  method: string,
): void {
  server.registerResource(
    name,
    new ResourceTemplate(uriTemplate, { list: undefined }),
    {
      title,
      description: `${title} as a concise read-only daemon snapshot.`,
      mimeType: "application/json",
    },
    async (uri, variables) =>
      resource(
        uri,
        await daemon.call(method, {
          projectId: String(variables.projectId),
        }),
      ),
  );
}

function resource(uri: URL, data: unknown) {
  return {
    contents: [
      {
        uri: uri.href,
        mimeType: "application/json",
        text: JSON.stringify(data, null, 2),
      },
    ],
  };
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}
