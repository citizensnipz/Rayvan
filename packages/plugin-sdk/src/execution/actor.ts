/**
 * Identity of the caller requesting a plugin capability execution.
 * Kept separate from service.ts so contracts can reference it without cycles.
 */
export type PluginExecutionActor =
  | {
      type: "user";
      id: string;
      displayName?: string;
    }
  | {
      type: "mcp_agent";
      id: string;
      displayName?: string;
    }
  | {
      type: "system";
      id: string;
    };
