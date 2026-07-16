#!/usr/bin/env node
import { parseCliArguments } from "./cli.js";
import { startRayvanMcpServer } from "./server/index.js";

let options: { clientId: string };
try {
  options = parseCliArguments(process.argv.slice(2));
} catch (error) {
  console.error(
    `[rayvan-mcp] ${error instanceof Error ? error.message : String(error)}`,
  );
  process.exit(2);
}

startRayvanMcpServer(options).catch((error: unknown) => {
  console.error("[rayvan-mcp] failed to start", error);
  process.exit(1);
});
