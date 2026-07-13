export interface AgentContext {
  workspaceId: string;
  projectId?: string;
  environmentId?: string;
}

export interface AgentToolDescriptor {
  name: string;
  description: string;
}

export interface AiAgent {
  id: string;
  name: string;
  availableTools: AgentToolDescriptor[];
}
