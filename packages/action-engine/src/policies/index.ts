export interface ActionPolicy {
  id: string;
  description: string;
  allowsExecution: boolean;
}

export const DEFAULT_ACTION_POLICIES: ActionPolicy[] = [
  {
    id: "require-human-approval",
    description: "All infrastructure mutations require explicit human approval.",
    allowsExecution: false,
  },
];
