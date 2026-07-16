export function parseCliArguments(
  argv: string[],
  env: NodeJS.ProcessEnv = process.env,
): { clientId: string } {
  if (argv[0] !== "serve") {
    throw new Error("Usage: rayvan-mcp serve --client-id <id>");
  }

  let clientId = env.RAYVAN_MCP_CLIENT_ID;
  for (let index = 1; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--client-id") {
      clientId = argv[index + 1];
      index += 1;
      continue;
    }
    throw new Error(`Unknown argument: ${argument}`);
  }
  if (!clientId) {
    throw new Error(
      "MCP client id is required via --client-id or RAYVAN_MCP_CLIENT_ID.",
    );
  }
  return { clientId };
}
