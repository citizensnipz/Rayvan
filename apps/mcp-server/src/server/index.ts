import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { createConsoleLogger } from "@rayvan/shared";
import { z } from "zod";
import { PLACEHOLDER_TOOL_NAMES } from "../tools/index.js";
import { MCP_SERVER_POLICIES } from "../policies/index.js";

const logger = createConsoleLogger("mcp-server");

export function createRayvanMcpServer(): McpServer {
  const server = new McpServer({
    name: "rayvan",
    version: "0.0.1",
  });

  for (const toolName of PLACEHOLDER_TOOL_NAMES) {
    server.registerTool(
      toolName,
      {
        description: `Placeholder Rayvan tool: ${toolName}`,
        inputSchema: z.object({}),
      },
      async () => ({
        content: [
          {
            type: "text" as const,
            text: `${toolName} is not implemented yet.`,
          },
        ],
        isError: true,
      }),
    );
  }

  logger.info("Rayvan MCP server initialized", MCP_SERVER_POLICIES);
  return server;
}

export async function startRayvanMcpServer(): Promise<void> {
  const server = createRayvanMcpServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
  logger.info("Rayvan MCP server connected via stdio");
}
