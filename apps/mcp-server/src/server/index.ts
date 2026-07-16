import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  DaemonClient,
  launchOrAttachDaemon,
  LocalClientCredentialStore,
} from "@rayvan/daemon-client";
import { registerRayvanPrompts } from "../prompts/index.js";
import { registerRayvanResources } from "../resources/index.js";
import { registerRayvanTools } from "../tools/index.js";

export type RayvanDaemonClient = Pick<DaemonClient, "call">;

export function createRayvanMcpServer(daemonClient: RayvanDaemonClient): McpServer {
  const server = new McpServer({
    name: "rayvan",
    version: "0.0.1",
  });

  registerRayvanTools(server, daemonClient);
  registerRayvanResources(server, daemonClient);
  registerRayvanPrompts(server);
  return server;
}

export interface StartRayvanMcpServerOptions {
  clientId: string;
  credentialStore?: LocalClientCredentialStore;
}

export async function startRayvanMcpServer(
  options: StartRayvanMcpServerOptions,
): Promise<void> {
  const credential = (
    options.credentialStore ?? new LocalClientCredentialStore()
  ).resolve(options.clientId);
  if (!credential) {
    throw new Error(`No local credential found for MCP client "${options.clientId}".`);
  }

  const { client: daemon } = await launchOrAttachDaemon({
    clientType: "mcp",
    clientVersion: "0.0.1",
    clientId: options.clientId,
    clientCredential: credential,
    daemonBinary: process.env.RAYVAN_DAEMON_BIN,
  });

  const server = createRayvanMcpServer(daemon);
  const transport = new StdioServerTransport();
  try {
    await server.connect(transport);
    console.error(`[rayvan-mcp] connected as ${options.clientId}`);
  } catch (error) {
    await daemon.close();
    throw error;
  }
}
