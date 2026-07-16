import {
  DaemonMethods,
  type ApprovalRequestRecord,
  type DaemonEvent,
  type DaemonStatus,
  type LocalClientPublicView,
  type LocalClientRecord,
  type McpApprovalPolicy,
  type OperationRecord,
  type BuiltInPermissionProfileId,
} from "@rayvan/daemon-contracts";

import { DaemonIpcTransport, type DaemonClientTransportOptions } from "./transport.js";

export class DaemonClient {
  private readonly transport: DaemonIpcTransport;

  constructor(options: DaemonClientTransportOptions) {
    this.transport = new DaemonIpcTransport(options);
  }

  async connect() {
    return this.transport.connect();
  }

  async close() {
    return this.transport.close();
  }

  onEvent(listener: (event: DaemonEvent) => void): () => void {
    return this.transport.onEvent(listener);
  }

  async subscribe(eventTypes?: string[]): Promise<void> {
    return this.transport.subscribe(eventTypes);
  }

  async call<T = unknown>(method: string, params?: unknown): Promise<T> {
    return (await this.transport.request(method, params)) as T;
  }

  // --- Typed helpers ---

  status(): Promise<DaemonStatus> {
    return this.call(DaemonMethods.status);
  }

  listProjects(params?: { includeArchived?: boolean }) {
    return this.call<unknown[]>(DaemonMethods.listProjects, params ?? {});
  }

  getProject(projectId: string) {
    return this.call(DaemonMethods.getProject, { projectId });
  }

  listEnvironments(projectId: string, options?: { includeArchived?: boolean }) {
    return this.call(DaemonMethods.listEnvironments, { projectId, ...options });
  }

  createEnvironment(params: Record<string, unknown>) {
    return this.call(DaemonMethods.createEnvironment, params);
  }

  updateEnvironment(environmentId: string, patch: Record<string, unknown>) {
    return this.call(DaemonMethods.updateEnvironment, {
      environmentId,
      ...patch,
    });
  }

  setConfigurationValue(params: Record<string, unknown>) {
    return this.call(DaemonMethods.setConfigurationValue, params);
  }

  setSensitiveConfigurationValue(params: Record<string, unknown>) {
    return this.call(DaemonMethods.setSensitiveConfigurationValue, params);
  }

  listFindings(params: Record<string, unknown>) {
    return this.call(DaemonMethods.listFindings, params);
  }

  getFinding(findingId: string) {
    return this.call(DaemonMethods.getFinding, { findingId });
  }

  scanFindings(params: Record<string, unknown>) {
    return this.call<OperationRecord>(DaemonMethods.scanFindings, params);
  }

  generatePlanFromFinding(params: Record<string, unknown>) {
    return this.call(DaemonMethods.generatePlanFromFinding, params);
  }

  approveChangePlan(params: Record<string, unknown>) {
    return this.call(DaemonMethods.approveChangePlan, params);
  }

  applyChangePlan(params: Record<string, unknown>) {
    return this.call<OperationRecord>(DaemonMethods.applyChangePlan, params);
  }

  verifyChangePlan(params: Record<string, unknown>) {
    return this.call<OperationRecord>(DaemonMethods.verifyChangePlan, params);
  }

  getOperation(operationId: string) {
    return this.call<OperationRecord>(DaemonMethods.getOperation, {
      operationId,
    });
  }

  listOperations(params?: Record<string, unknown>) {
    return this.call<OperationRecord[]>(DaemonMethods.listOperations, params ?? {});
  }

  listApprovals(params?: Record<string, unknown>) {
    return this.call<ApprovalRequestRecord[]>(
      DaemonMethods.listApprovals,
      params ?? {},
    );
  }

  decideApproval(params: {
    approvalId: string;
    decision: "approved" | "denied";
    rememberScope?: boolean;
  }) {
    return this.call<ApprovalRequestRecord>(DaemonMethods.decideApproval, params);
  }

  createMcpClient(params: {
    name: string;
    permissionProfileId: BuiltInPermissionProfileId;
    projectScopes: string[];
    environmentScopes?: string[];
    approvalPolicy?: McpApprovalPolicy;
  }) {
    return this.call<{
      client: LocalClientRecord;
      /** One-time credential for secure local storage — never logged. */
      credential: string;
      mcpConfig: Record<string, unknown>;
    }>(DaemonMethods.createMcpClient, params);
  }

  listMcpClients() {
    return this.call<LocalClientPublicView[]>(DaemonMethods.listMcpClients);
  }

  revokeMcpClient(clientId: string) {
    return this.call(DaemonMethods.revokeMcpClient, { clientId });
  }

  getMcpClientScope() {
    return this.call(DaemonMethods.getMcpClientScope);
  }
}
