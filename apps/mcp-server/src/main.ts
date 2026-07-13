#!/usr/bin/env node
import { startRayvanMcpServer } from "./server/index.js";

startRayvanMcpServer().catch((error: unknown) => {
  console.error("[rayvan-mcp] failed to start", error);
  process.exit(1);
});
