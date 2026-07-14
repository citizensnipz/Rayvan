export const manifest = {
  id: "github",
  name: "GitHub",
  version: "0.0.1",
  protocolVersion: "1",
  capabilities: [
    "resource-discovery",
    "configuration-read",
    "health-read",
    "action-plan",
    "action-execute",
  ],
} as const;
