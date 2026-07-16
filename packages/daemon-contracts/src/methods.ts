/** Stable daemon method names (not MCP tool names). */
export const DaemonMethods = {
  // System / handshake
  handshake: "system.handshake",
  status: "system.status",
  diagnostics: "system.diagnostics",
  subscribe: "system.subscribe",
  ping: "system.ping",
  shutdown: "system.shutdown",

  // Projects
  listProjects: "projects.list",
  getProject: "projects.get",
  getProjectOverview: "projects.getOverview",
  createProject: "projects.create",
  updateProject: "projects.update",

  // Environments
  listEnvironments: "environments.list",
  getEnvironment: "environments.get",
  createEnvironment: "environments.create",
  updateEnvironment: "environments.update",
  archiveEnvironment: "environments.archive",
  compareEnvironments: "environments.compare",
  getEnvironmentConfiguration: "environments.getConfiguration",
  getEnvironmentResources: "environments.getResources",

  // Integrations
  listIntegrations: "integrations.list",
  getIntegration: "integrations.get",
  listIntegrationResources: "integrations.listResources",
  getIntegrationHealth: "integrations.getHealth",
  syncProject: "integrations.syncProject",
  syncEnvironment: "integrations.syncEnvironment",
  syncIntegration: "integrations.syncIntegration",
  inspectResource: "integrations.inspectResource",

  // Configuration
  listConfigurationKeys: "configuration.listKeys",
  getConfigurationKey: "configuration.getKey",
  findConfigurationUsage: "configuration.findUsage",
  getConfigurationStatus: "configuration.getStatus",
  listUnmanagedConfiguration: "configuration.listUnmanaged",
  setConfigurationValue: "configuration.setValue",
  setSensitiveConfigurationValue: "configuration.setSensitiveValue",
  clearConfigurationValue: "configuration.clearValue",
  setConfigurationMetadata: "configuration.setMetadata",
  setConfigurationTargets: "configuration.setTargets",
  removeConfigurationTarget: "configuration.removeTarget",
  adoptDiscoveredConfiguration: "configuration.adoptDiscovered",
  ignoreDiscoveredConfiguration: "configuration.ignoreDiscovered",
  revealSensitiveConfigurationValue: "configuration.revealSensitiveValue",

  // Findings
  listFindings: "findings.list",
  getFinding: "findings.get",
  explainFinding: "findings.explain",
  getFindingSummary: "findings.getSummary",
  scanFindings: "findings.scan",
  acknowledgeFinding: "findings.acknowledge",
  dismissFinding: "findings.dismiss",
  suppressFinding: "findings.suppress",
  reopenFinding: "findings.reopen",

  // Change plans
  listChangePlans: "changePlans.list",
  getChangePlan: "changePlans.get",
  generateChangePlan: "changePlans.generate",
  generatePlanFromFinding: "changePlans.generateFromFinding",
  rejectChangePlan: "changePlans.reject",
  approveChangePlan: "changePlans.approve",
  applyChangePlan: "changePlans.apply",
  verifyChangePlan: "changePlans.verify",
  retryFailedChange: "changePlans.retryFailed",

  // Operations / approvals
  listOperations: "operations.list",
  getOperation: "operations.get",
  cancelOperation: "operations.cancel",
  listApprovals: "approvals.list",
  decideApproval: "approvals.decide",

  // MCP clients
  listMcpClients: "mcpClients.list",
  getMcpClient: "mcpClients.get",
  createMcpClient: "mcpClients.create",
  updateMcpClient: "mcpClients.update",
  revokeMcpClient: "mcpClients.revoke",
  rotateMcpClientCredential: "mcpClients.rotateCredential",
  getMcpClientScope: "mcpClients.getScope",
  listAvailableCapabilities: "mcpClients.listCapabilities",
  listMcpAuditEvents: "mcpClients.listAuditEvents",

  // Plugins
  listPlugins: "plugins.list",
  listPluginActions: "plugins.listActions",
} as const;

export type DaemonMethod = (typeof DaemonMethods)[keyof typeof DaemonMethods];
