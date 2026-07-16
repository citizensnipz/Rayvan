import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

export const RAYVAN_PROMPT_NAMES = [
  "diagnose_project",
  "diagnose_production",
  "compare_environments",
  "review_configuration_drift",
  "plan_finding_resolution",
  "review_recent_operations",
] as const;

export function registerRayvanPrompts(server: McpServer): void {
  projectPrompt(
    server,
    "diagnose_project",
    "Diagnose project",
    "Review the project's environments, integrations, configuration status, and open Findings. Report evidence before recommendations.",
  );
  projectPrompt(
    server,
    "diagnose_production",
    "Diagnose production",
    "Identify the most important Production problems from environments, integrations, configuration drift, Findings, and recent failed operations. Do not mutate state.",
  );

  server.registerPrompt(
    "compare_environments",
    {
      title: "Compare environments",
      description: "Compare configuration and resources between two environments.",
      argsSchema: {
        project_id: z.string().min(1),
        left_environment_id: z.string().min(1),
        right_environment_id: z.string().min(1),
      },
    },
    ({ project_id, left_environment_id, right_environment_id }) =>
      prompt(
        `Compare environments ${left_environment_id} and ${right_environment_id} in Rayvan project ${project_id}. Review desired configuration, observed resources, and Findings. Do not mutate state.`,
      ),
  );

  server.registerPrompt(
    "review_configuration_drift",
    {
      title: "Review configuration drift",
      description:
        "Identify missing, mismatched, remote-changed, and unapplied configuration.",
      argsSchema: {
        project_id: z.string().min(1),
        environment_id: z.string().min(1).optional(),
      },
    },
    ({ project_id, environment_id }) =>
      prompt(
        `Review configuration drift in Rayvan project ${project_id}${environment_id ? ` for environment ${environment_id}` : ""}. Identify missing, mismatched, remote-changed, unmanaged, and unapplied values. Do not reveal plaintext secrets.`,
      ),
  );

  server.registerPrompt(
    "plan_finding_resolution",
    {
      title: "Plan Finding resolution",
      description:
        "Review a Finding and propose a safe daemon-generated remediation plan.",
      argsSchema: { finding_id: z.string().min(1) },
    },
    ({ finding_id }) =>
      prompt(
        `Inspect Finding ${finding_id}, explain its evidence, and generate a structured change plan through Rayvan. Review freshness, scope, and risk before requesting approval or applying it.`,
      ),
  );

  projectPrompt(
    server,
    "review_recent_operations",
    "Review recent operations",
    "Explain recent failed or partially successful operations, their safe errors, affected scopes, and the next safe action.",
  );
}

function projectPrompt(
  server: McpServer,
  name: string,
  title: string,
  instruction: string,
): void {
  server.registerPrompt(
    name,
    {
      title,
      description: instruction,
      argsSchema: { project_id: z.string().min(1) },
    },
    ({ project_id }) => prompt(`${instruction} Project id: ${project_id}.`),
  );
}

function prompt(text: string) {
  return {
    messages: [
      {
        role: "user" as const,
        content: { type: "text" as const, text },
      },
    ],
  };
}
