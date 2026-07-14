export const manifest = {
  id: "runpod",
  name: "RunPod",
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