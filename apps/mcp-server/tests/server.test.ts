import { describe, expect, it } from "vitest";
import { PLACEHOLDER_TOOL_NAMES } from "../src/tools/index.js";
import { MCP_SERVER_POLICIES } from "../src/policies/index.js";

describe("@rayvan/mcp-server", () => {
  it("registers placeholder tools", () => {
    expect(PLACEHOLDER_TOOL_NAMES).toContain("list_projects");
    expect(PLACEHOLDER_TOOL_NAMES.length).toBeGreaterThan(0);
  });

  it("enforces MCP safety policies", () => {
    expect(MCP_SERVER_POLICIES.exposeRawSecrets).toBe(false);
    expect(MCP_SERVER_POLICIES.bypassActionApproval).toBe(false);
  });
});
