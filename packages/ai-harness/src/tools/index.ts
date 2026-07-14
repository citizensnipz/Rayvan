export interface HarnessTool {
  name: string;
  description: string;
  invoke(input: Record<string, unknown>): Promise<unknown>;
}

export const PLACEHOLDER_TOOLS: HarnessTool[] = [];
