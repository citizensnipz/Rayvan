import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type CSSProperties,
  type FormEvent,
} from "react";
import type {
  ApprovalRequestRecord,
  DaemonDiagnostics,
  LocalClientPublicView,
  OperationRecord,
} from "@rayvan/daemon-contracts";
import { Button } from "@rayvan/ui";

import {
  desktopDaemon,
  useDaemonConnection,
} from "../../lib/daemon/index.js";
import { AgentMcpTabBar } from "./AgentMcpTabBar.js";
import {
  mapActivityRows,
  mapApprovalRows,
  mapCapabilities,
  mapClientRows,
  mapDaemonView,
  mapOverview,
  mapSetupView,
} from "./mappers.js";
import { tabPanelId } from "./tab-ids.js";
import type { AgentMcpTabId } from "./view-models.js";

const pageStyle: CSSProperties = {
  padding: "1.25rem 1.5rem",
};

const cardGridStyle: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(12rem, 1fr))",
  gap: "0.75rem",
  marginBottom: "1rem",
};

const cardStyle: CSSProperties = {
  border: "1px solid var(--color-border)",
  borderRadius: "8px",
  padding: "0.85rem 1rem",
  background: "var(--color-surface)",
};

const tableStyle: CSSProperties = {
  width: "100%",
  borderCollapse: "collapse",
  fontSize: "0.9rem",
};

const thStyle: CSSProperties = {
  textAlign: "left",
  padding: "0.5rem 0.4rem",
  borderBottom: "1px solid var(--color-border)",
  color: "var(--color-text-muted)",
  fontWeight: 600,
};

const tdStyle: CSSProperties = {
  padding: "0.55rem 0.4rem",
  borderBottom: "1px solid var(--color-border)",
  verticalAlign: "top",
};

const bannerStyle: CSSProperties = {
  marginBottom: "1rem",
  padding: "0.65rem 1rem",
  borderRadius: "8px",
  border: "1px solid var(--color-border-strong)",
  background: "var(--color-surface-muted)",
};

const preStyle: CSSProperties = {
  margin: 0,
  padding: "0.85rem 1rem",
  borderRadius: "8px",
  border: "1px solid var(--color-border)",
  background: "var(--color-surface-muted)",
  overflowX: "auto",
  fontSize: "0.85rem",
  whiteSpace: "pre-wrap",
};

export function AgentMcpWorkspace() {
  const { connected, snapshot, status, refresh, reconnect } =
    useDaemonConnection();
  const [activeTab, setActiveTab] = useState<AgentMcpTabId>("overview");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [clients, setClients] = useState<LocalClientPublicView[]>([]);
  const [approvals, setApprovals] = useState<ApprovalRequestRecord[]>([]);
  const [operations, setOperations] = useState<OperationRecord[]>([]);
  const [auditEvents, setAuditEvents] = useState<
    Array<Record<string, unknown>>
  >([]);
  const [capabilities, setCapabilities] = useState<{
    permissions: string[];
    methods: string[];
  }>({ permissions: [], methods: [] });
  const [diagnostics, setDiagnostics] = useState<DaemonDiagnostics | null>(
    null,
  );
  const [createName, setCreateName] = useState("");
  const [createProfile, setCreateProfile] = useState("operator");
  const [createdClientId, setCreatedClientId] = useState<string | null>(null);
  const [banner, setBanner] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!connected) {
      setClients([]);
      setApprovals([]);
      setOperations([]);
      setAuditEvents([]);
      setCapabilities({ permissions: [], methods: [] });
      setDiagnostics(null);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const [
        nextClients,
        nextApprovals,
        nextOperations,
        nextAudit,
        nextCapabilities,
        nextDiagnostics,
      ] = await Promise.all([
        desktopDaemon.listMcpClients(),
        desktopDaemon.listApprovals({ status: "pending" }),
        desktopDaemon.listOperations(),
        desktopDaemon.listMcpAuditEvents(50),
        desktopDaemon.listAvailableCapabilities(),
        desktopDaemon.diagnostics().catch(() => null),
      ]);
      setClients(nextClients);
      setApprovals(nextApprovals);
      setOperations(nextOperations.slice(0, 40));
      setAuditEvents(
        Array.isArray(nextAudit)
          ? (nextAudit as Array<Record<string, unknown>>)
          : [],
      );
      setCapabilities(nextCapabilities);
      setDiagnostics(nextDiagnostics);
    } catch (loadError) {
      setError(
        loadError instanceof Error
          ? loadError.message
          : "Failed to load Agent / MCP workspace",
      );
    } finally {
      setLoading(false);
    }
  }, [connected]);

  useEffect(() => {
    void load();
  }, [load]);

  const overview = useMemo(
    () =>
      mapOverview({
        clients,
        approvals,
        operations,
        snapshot,
        status,
      }),
    [clients, approvals, operations, snapshot, status],
  );
  const clientRows = useMemo(() => mapClientRows(clients), [clients]);
  const approvalRows = useMemo(() => mapApprovalRows(approvals), [approvals]);
  const activityRows = useMemo(
    () =>
      mapActivityRows({
        operations,
        auditEvents: auditEvents as never,
      }),
    [operations, auditEvents],
  );
  const capabilityView = useMemo(
    () => mapCapabilities(capabilities),
    [capabilities],
  );
  const daemonView = useMemo(
    () => mapDaemonView({ snapshot, status, diagnostics }),
    [snapshot, status, diagnostics],
  );
  const setupView = useMemo(
    () =>
      mapSetupView(
        createdClientId ??
          clients.find((client) => client.type === "mcp")?.id,
      ),
    [createdClientId, clients],
  );

  async function handleCreateClient(event: FormEvent) {
    event.preventDefault();
    if (!createName.trim()) {
      return;
    }
    setBanner(null);
    try {
      const result = (await desktopDaemon.createMcpClient({
        name: createName.trim(),
        permissionProfileId: createProfile,
        projectScopes: [],
      })) as {
        client: { id: string };
        credential?: string;
      };
      setCreatedClientId(result.client.id);
      setCreateName("");
      setBanner(
        `Created MCP client ${result.client.id}. Credential stored in OS keyring — use the client id in host config.`,
      );
      await load();
    } catch (createError) {
      setBanner(
        createError instanceof Error
          ? createError.message
          : "Failed to create MCP client",
      );
    }
  }

  async function handleRevoke(clientId: string) {
    await desktopDaemon.revokeMcpClient(clientId);
    setBanner(`Revoked ${clientId}`);
    await load();
  }

  async function handleRotate(clientId: string) {
    await desktopDaemon.rotateMcpClientCredential(clientId);
    setBanner(
      `Rotated credential for ${clientId}. New secret is in the OS keyring only.`,
    );
    await load();
  }

  async function handleDecide(
    approvalId: string,
    decision: "approved" | "denied",
  ) {
    await desktopDaemon.decideApproval({ approvalId, decision });
    setBanner(`Approval ${decision}`);
    await load();
  }

  return (
    <div style={pageStyle}>
      <header style={{ marginBottom: "1rem" }}>
        <h2 style={{ margin: "0 0 0.35rem" }}>Agent / MCP</h2>
        <p style={{ margin: 0, color: "var(--color-text-muted)" }}>
          Manage local MCP clients, approvals, and daemon health for Rayvan.
        </p>
      </header>

      {!connected ? (
        <div role="status" style={bannerStyle}>
          Daemon is offline. Connect rayvand to manage MCP clients and
          approvals.{" "}
          <Button onClick={() => void reconnect()}>Retry connection</Button>
        </div>
      ) : null}

      {banner ? (
        <div role="status" style={bannerStyle}>
          {banner}
        </div>
      ) : null}
      {error ? (
        <div role="alert" style={bannerStyle}>
          {error}
        </div>
      ) : null}

      <AgentMcpTabBar activeTab={activeTab} onSelect={setActiveTab} />

      <div
        role="tabpanel"
        id={tabPanelId(activeTab)}
        aria-labelledby={`agent-mcp-tab-${activeTab}`}
      >
        {loading ? <p>Loading…</p> : null}

        {activeTab === "overview" ? (
          <div style={cardGridStyle}>
            <StatCard label="MCP clients" value={String(overview.mcpClientCount)} />
            <StatCard
              label="Active MCP"
              value={String(overview.activeMcpClientCount)}
            />
            <StatCard
              label="Pending approvals"
              value={String(overview.pendingApprovalCount)}
            />
            <StatCard
              label="Recent operations"
              value={String(overview.recentOperationCount)}
            />
            <StatCard
              label="Daemon"
              value={
                overview.daemonConnected
                  ? overview.daemonState ?? "connected"
                  : "offline"
              }
            />
            <StatCard
              label="Version"
              value={overview.daemonVersion ?? "—"}
            />
          </div>
        ) : null}

        {activeTab === "clients" ? (
          <section>
            <form
              onSubmit={(event) => void handleCreateClient(event)}
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: "0.5rem",
                marginBottom: "1rem",
                alignItems: "end",
              }}
            >
              <label style={{ display: "grid", gap: "0.25rem" }}>
                <span>Name</span>
                <input
                  value={createName}
                  onChange={(event) => setCreateName(event.target.value)}
                  placeholder="Cursor / Claude Desktop"
                />
              </label>
              <label style={{ display: "grid", gap: "0.25rem" }}>
                <span>Profile</span>
                <select
                  value={createProfile}
                  onChange={(event) => setCreateProfile(event.target.value)}
                >
                  <option value="read_only">Read only</option>
                  <option value="planner">Planner</option>
                  <option value="operator">Operator</option>
                </select>
              </label>
              <Button type="submit" disabled={!connected}>
                Create client
              </Button>
              <Button type="button" onClick={() => void load()}>
                Refresh
              </Button>
            </form>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Name</th>
                  <th style={thStyle}>Type</th>
                  <th style={thStyle}>Status</th>
                  <th style={thStyle}>Profile</th>
                  <th style={thStyle}>Scope</th>
                  <th style={thStyle}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {clientRows.map((row) => (
                  <tr key={row.id}>
                    <td style={tdStyle}>
                      <div>{row.name}</div>
                      <code style={{ fontSize: "0.8rem" }}>{row.id}</code>
                    </td>
                    <td style={tdStyle}>{row.type}</td>
                    <td style={tdStyle}>
                      {row.status}
                      {row.connected ? " · connected" : ""}
                    </td>
                    <td style={tdStyle}>{row.profile}</td>
                    <td style={tdStyle}>{row.projectScopeLabel}</td>
                    <td style={tdStyle}>
                      {row.type === "mcp" && row.status === "active" ? (
                        <div style={{ display: "flex", gap: "0.35rem" }}>
                          <Button onClick={() => void handleRotate(row.id)}>
                            Rotate
                          </Button>
                          <Button onClick={() => void handleRevoke(row.id)}>
                            Revoke
                          </Button>
                        </div>
                      ) : (
                        "—"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        ) : null}

        {activeTab === "tools" ? (
          <section
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "1rem",
            }}
          >
            <div>
              <h3>Permissions</h3>
              <ul>
                {capabilityView.permissions.map((permission) => (
                  <li key={permission}>
                    <code>{permission}</code>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h3>Daemon methods</h3>
              <ul style={{ maxHeight: "24rem", overflow: "auto" }}>
                {capabilityView.methods.map((method) => (
                  <li key={method}>
                    <code>{method}</code>
                  </li>
                ))}
              </ul>
            </div>
          </section>
        ) : null}

        {activeTab === "approvals" ? (
          <section>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Summary</th>
                  <th style={thStyle}>Type</th>
                  <th style={thStyle}>Status</th>
                  <th style={thStyle}>Created</th>
                  <th style={thStyle}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {approvalRows.length === 0 ? (
                  <tr>
                    <td style={tdStyle} colSpan={5}>
                      No pending approvals.
                    </td>
                  </tr>
                ) : (
                  approvalRows.map((row) => (
                    <tr key={row.id}>
                      <td style={tdStyle}>{row.summary}</td>
                      <td style={tdStyle}>{row.type}</td>
                      <td style={tdStyle}>{row.status}</td>
                      <td style={tdStyle}>{row.createdAtLabel}</td>
                      <td style={tdStyle}>
                        {row.status === "pending" ? (
                          <div style={{ display: "flex", gap: "0.35rem" }}>
                            <Button
                              onClick={() => void handleDecide(row.id, "approved")}
                            >
                              Approve
                            </Button>
                            <Button
                              onClick={() => void handleDecide(row.id, "denied")}
                            >
                              Deny
                            </Button>
                          </div>
                        ) : (
                          "—"
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </section>
        ) : null}

        {activeTab === "activity" ? (
          <section>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Kind</th>
                  <th style={thStyle}>Title</th>
                  <th style={thStyle}>Status</th>
                  <th style={thStyle}>When</th>
                </tr>
              </thead>
              <tbody>
                {activityRows.length === 0 ? (
                  <tr>
                    <td style={tdStyle} colSpan={4}>
                      No recent activity.
                    </td>
                  </tr>
                ) : (
                  activityRows.map((row) => (
                    <tr key={`${row.kind}-${row.id}`}>
                      <td style={tdStyle}>{row.kind}</td>
                      <td style={tdStyle}>{row.title}</td>
                      <td style={tdStyle}>{row.status ?? "—"}</td>
                      <td style={tdStyle}>{row.timestampLabel}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </section>
        ) : null}

        {activeTab === "setup" ? (
          <section>
            <p>
              Configure an MCP host with client id{" "}
              <code>{setupView.exampleClientId}</code>. The credential remains
              in the OS keyring.
            </p>
            <pre style={preStyle}>{setupView.commandSnippet}</pre>
            <ul>
              {setupView.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          </section>
        ) : null}

        {activeTab === "daemon" ? (
          <section>
            <div style={cardGridStyle}>
              <StatCard
                label="Connected"
                value={daemonView.connected ? "yes" : "no"}
              />
              <StatCard label="State" value={daemonView.state ?? "—"} />
              <StatCard
                label="Clients"
                value={String(daemonView.connectedClients ?? "—")}
              />
              <StatCard
                label="Pending approvals"
                value={String(daemonView.pendingApprovals ?? "—")}
              />
              <StatCard
                label="Active ops"
                value={String(daemonView.activeOperations ?? "—")}
              />
              <StatCard
                label="Plugin host"
                value={daemonView.pluginHostStatus ?? "—"}
              />
            </div>
            <p>
              Endpoint: <code>{daemonView.endpoint || "—"}</code>
            </p>
            <p>
              Session: <code>{daemonView.sessionId ?? "—"}</code>
            </p>
            <p>
              Authenticated as:{" "}
              <code>{daemonView.authenticatedClientId ?? "—"}</code>
            </p>
            {daemonView.lastError ? (
              <p role="status">Last error: {daemonView.lastError}</p>
            ) : null}
            <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem" }}>
              <Button onClick={() => void refresh()}>Refresh status</Button>
              <Button onClick={() => void reconnect()}>Reconnect</Button>
              <Button onClick={() => void load()}>Reload diagnostics</Button>
            </div>
            {daemonView.diagnosticsJson ? (
              <pre style={preStyle}>{daemonView.diagnosticsJson}</pre>
            ) : null}
          </section>
        ) : null}
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div style={cardStyle}>
      <div style={{ color: "var(--color-text-muted)", fontSize: "0.8rem" }}>
        {label}
      </div>
      <div style={{ fontSize: "1.25rem", fontWeight: 600 }}>{value}</div>
    </div>
  );
}
